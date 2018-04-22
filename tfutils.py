"""Utilities for TensorFlow."""

import os
import re
import logging
import numpy as np
import scipy.signal
import tensorflow as tf

import six
from tensorflow.python.training import session_run_hook
from tensorflow.python.framework import ops
from tensorflow.python.training.session_run_hook import SessionRunArgs

from logging import info


def my_formatter(tensor_values):
    """Output formatter for LoggingTensorHook that logs the message
to our logger as a side effect."""
    stats = []
    for tag in tensor_values.keys():
        stats.append("%s = %s" % (tag, tensor_values[tag]))
    s = 'logger: %s' % ", ".join(stats)
    logging.info(s)
    return s


def _as_graph_element(obj):
    """Retrieves Graph element."""
    graph = ops.get_default_graph()
    if not isinstance(obj, six.string_types):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute "
                             "that is equal to current graph %s."
                             % (obj, graph))
        return obj
    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, as this `Operation` "
                             "has multiple outputs (at least 2)." % obj)
    return element


class MyLoggingTensorHook(session_run_hook.SessionRunHook):
    """Prints the given tensors every N local steps, every N seconds, or at
    end.  The tensors will be printed to the log, with `INFO` severity. If you
    are not seeing the logs, you might want to add the following line after
    your imports:
    ```python
        tf.logging.set_verbosity(tf.logging.INFO)
    ```
    Note that if `at_end` is True, `tensors` should not include any tensor
    whose evaluation produces a side effect such as consuming additional
    inputs.
    """

    def __init__(self, tensors, every_n_iter=None, every_n_secs=None,
                 at_end=False, formatter=None, steps_per_epoch=None):
        """Initializes a `LoggingTensorHook`.
        Args:
            tensors: `dict` that maps string-valued tags to tensors/tensor names,
                    or `iterable` of tensors/tensor names.
            every_n_iter: `int`, print the values of `tensors` once every N local
                    steps taken on the current worker.
            every_n_secs: `int` or `float`, print the values of `tensors` once every N
                    seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
                    provided.
            at_end: `bool` specifying whether to print the values of `tensors` at the
                    end of the run.
            formatter: function, takes dict of `tag`->`Tensor` and returns a string.
                    If `None` uses default printing all tensors.
        Raises:
            ValueError: if `every_n_iter` is non-positive.
        """
        only_log_at_end = (
                at_end and (every_n_iter is None) and (every_n_secs is None))
        if (not only_log_at_end and
                (every_n_iter is None) == (every_n_secs is None)):
            raise ValueError(
                    "either at_end and/or exactly one of every_n_iter and every_n_secs "
                    "must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)
        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}
        else:
            self._tag_order = tensors.keys()
        self._tensors = tensors
        self._formatter = formatter
        self._timer = (
                tf.train.NeverTriggerTimer() if only_log_at_end else
                tf.train.SecondOrStepTimer(every_secs=every_n_secs,
                                           every_steps=every_n_iter))
        self._log_at_end = at_end
        self._steps_per_epoch = steps_per_epoch
        self.global_step = 0

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        # Convert names to tensors if given
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}
        # self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        # if self._global_step_tensor is None:
        #     raise RuntimeError("global step should be created to use stopatstephook.")
        # self._current_tensors['global_step'] = self._global_step_tensor
        self._current_tensors['global_step'] = tf.train.get_global_step()

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)
        if self._should_trigger:
            return SessionRunArgs(self._current_tensors)
        else:
            return None

    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(
            self._iter_count)
        if self._formatter:
            logging.info(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                stats.append("%s = %s" % (tag, tensor_values[tag]))
            global_step = tensor_values['global_step']
            stats.append("global_step = %s" % global_step)
            if self._steps_per_epoch is not None:
                epochs = (global_step + self._steps_per_epoch - 1) // \
                         self._steps_per_epoch
                stats.append("epochs = %s" % epochs)
            if elapsed_secs is not None:
                logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
            else:
                logging.info("%s", ", ".join(stats))
        np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        # _ = run_context
        if self._should_trigger:
            self._log_tensors(run_values.results)
            self.global_step = run_values.results['global_step']

        self._iter_count += 1

    def end(self, session):
        if self._log_at_end:
            values = session.run(self._current_tensors)
            self._log_tensors(values)


def simplify_event_tag(tag):
    """Remove suffixes like _1, _2 from tag names."""
    return re.sub(r'_\d+$', r'', tag)


def read_events_file(filename):
    """Read summaries from the given events file, with the given tag.
Returns an array of (step, value) tuples."""
    info('Reading events file %s' % filename)
    events = {}
    for e in tf.train.summary_iterator(filename):
        for v in e.summary.value:
            if v.tag not in events:
                events[v.tag] = []
            events[v.tag].append((e.step, v.simple_value))

    info('Found tags: %s' % ', '.join(events.keys()))
    return events


def get_events(events, simple_tag):
    """Get the events for the corresponding tag, removing suffixes like _1
from the tag name if possible."""
    matching_tags = set()
    for event_tag in events.keys():
        if re.match(simple_tag + r'(_\d+)?$', event_tag):
            matching_tags.add(event_tag)

    if len(matching_tags) > 1:
        if simple_tag in matching_tags:
            return events[simple_tag]
        else:
            raise KeyError('Too many tags match %s' % simple_tag)
    elif len(matching_tags) == 1:
        return events[matching_tags.pop()]
    else:
        raise KeyError('Cannot find tag %s' % simple_tag)


def _find_event_files(model_dir):
    train_events_file = None
    val_events_file = None

    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                filename = os.path.join(root, file)
                if root.endswith('/eval'):
                    if val_events_file is not None:
                        raise RuntimeError(
                            'Found multiple val event files: %s and %s' %
                            (val_events_file, filename))
                    logging.info('Validation file: %s', filename)
                    val_events_file = filename
                else:
                    if train_events_file is not None:
                        raise RuntimeError(
                            'Found multiple train event files: %s and %s' %
                            (train_events_file, filename))
                    logging.info('Train file: %s', filename)
                    train_events_file = filename

    if train_events_file is None:
        raise RuntimeError('Cannot find train events file')
    # if val_events_file is None:
    #     raise RuntimeError('Cannot find validation events file')

    return train_events_file, val_events_file


def read_train_eval_events(model_dir):
    """Read the loss summaries for the given model.
Returns two arrays of (step, loss), for training and validation."""
    if not os.path.exists(model_dir):
        raise RuntimeError('Model directory does not exist: %s'
                           % model_dir)
    train_events_file, val_events_file = _find_event_files(model_dir)
    train_events = read_events_file(train_events_file)

    if val_events_file is None:
        logging.warn('Did not found validation events file')
        val_events = None
    else:
        val_events = read_events_file(val_events_file)

    return train_events, val_events


class TensorHistoryHook(tf.train.SessionRunHook):
    """Collect a history of a given tensor's values."""
    def __init__(self, name, every_n_steps=1):
        """name = name of the tensor to collect."""
        super().__init__()
        self.name = name
        self.every_n_steps = every_n_steps

    @property
    def should_run(self):
        return self.step % self.every_n_steps == 0

    def begin(self):
        self.steps = []
        self.values = []
        self.step = 0

    def before_run(self, run_context):
        if self.should_run:
            graph = run_context.session.graph
            tensor = graph.get_tensor_by_name(self.name)
            return tf.train.SessionRunArgs(fetches=tensor)
        else:
            return None

    def after_run(self, run_context, run_values):
        if self.should_run:
            value = run_values.results
            self.values.append(value)
            self.steps.append(self.step)
        self.step += 1

    def end(self, sess):
        self.values = np.asarray(self.values)


class StopRequestHook(tf.train.SessionRunHook):
    """Requests training to step at fixed intervals."""
    def __init__(self, every_n_steps=None, every_n_secs=None):
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('Exactly one of every_n_steps and every_n_secs '
                             'should be provided.')
        self._timer = tf.train.SecondOrStepTimer(every_secs=every_n_secs,
                                                 every_steps=every_n_steps)
        self.global_step = 0

    def begin(self):
        # info('StopRequestHook: begin')
        self._tensors = {}
        self._tensors['global_step'] = tf.train.get_global_step()

    def before_run(self, run_context):
        # info('StopRequestHook: before_run')
        return SessionRunArgs(self._tensors)

    def after_run(self, run_context, run_values):
        # info('StopRequestHook: after_run')
        self.global_step = run_values.results['global_step']
        # info('New global step: %d' % self.global_step)
        if self._timer.should_trigger_for_step(self.global_step):
            info('StopRequestHook: Requesting stop')
            self._timer.update_last_triggered_step(self.global_step)
            run_context.request_stop()


def smooth_data_series(data, window=51, poly_order=3):
    """Smooth the y values of a given series of (x, y) values.
The parameters are passed on to a Savitzky-Golay filter."""
    return scipy.signal.savgol_filter(data, window, poly_order, axis=0)
