#!/usr/bin/env python3

"""Regression experiments with deep networks.

Usage: deep_regression.py [options]

Options:
  -h, --help            Show this.
  --omega OMEGA         Target function (sine) frequency [default: 3.0]
  --noise-sigma NOISE_SIGMA
                        Target function (sine) noise std-dev [default: 0.2]
  --xmin XMIN           x is sampled from [xmin, xmax] [default: 0.]
  --xmax XMAX           x is sampled from [xmin, xmax] [default: 2.*pi]
  --train               Train the model
  --train-and-eval      Train and evaluate the model
  --widths WIDTHS       Hidden layer widths (commma-separated)
                        [default: 100,100]
  --n-train N_TRAIN     Number of training samples [default: 128]
  --n-val N_VAL         Number of validation samples [default: 128]
  --batch-size BATCH_SIZE
                        Mini-batch size [default: 32]
  --epochs EPOCHS       Number of training epochs [default: 200]
  --learning-rate LEARNING_RATE
                        Learning rate [default: 0.001]
  --eval-every-steps EVAL_EVERY_STEPS
                        Frequency of evaluating the model during training.
                        Equal to the steps between checkpoints. [default: 1000]
  --save-summary-steps SAVE_SUMMARY_STEPS
                        Frequency of saving summaries. [default: 100]
  --save-eval-plots-every-steps SAVE_EVAL_PLOTS_STEPS
                        Frequency of saving summaries. [default: 1000]
  --delete-saved-model  Delete model directory (checkpoints, events, etc.)
                        before starting
  --kinks-initializer   Use a kinks-based initializer for the first layer.
"""

import os
import sys
import time
import math
import shutil
import collections
import numpy as np
import tensorflow as tf
import logging
import tfutils

from tensorflow.python.training.session_run_hook import SessionRunArgs

# Use backend that doesn't open a window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from logging import info
from docopt import docopt

if sys.version_info[0] < 3:
    raise Exception('Must be using Python 3')

ExperimentParams = collections.namedtuple(
    'ExperimentParams',
    ['model_name', 'model_fn', 'xmin', 'xmax', 'omega', 'noise_sigma',
     'widths',
     'n_train', 'n_val',
     'learning_rate', 'batch_size', 'epochs',
     'save_summary_steps', 'save_checkpoint_steps',
     'save_eval_plots_steps'])


def create_target_fn(noise_sigma, omega=2.):
    def target_fn(x):
        return np.sin(omega*x) + noise_sigma * np.random.randn(*(x.shape))
    return target_fn


def save_weight_norm_summaries(layer, layer_idx):
    weights = tf.get_default_graph().get_tensor_by_name(
        os.path.split(layer.name)[0] + '/kernel:0')
    biases = tf.get_default_graph().get_tensor_by_name(
        os.path.split(layer.name)[0] + '/bias:0')
    weights_norm = tf.norm(weights)
    biases_norm = tf.norm(biases)
    tf.summary.scalar('weights%d' % layer_idx, weights_norm)
    tf.summary.scalar('biases%d' % layer_idx, biases_norm)


def get_kinks_initializer(num_kinks, xmin, xmax):
    kinks = xmin + (xmax - xmin) * np.random.rand(num_kinks)
    weights = np.random.randn(num_kinks)
    biases = - weights * kinks
    return tf.constant_initializer(weights), tf.constant_initializer(biases)


def create_model_fn(activation, kinks_initializer):
    def model_fn(features, labels, mode, params):
        """Create a fully-connected regression model. Hidden layers with relu
    activation lead to the output."""
        i = 1
        outputs = features['x']

        for width in params['widths']:
            name = 'hidden%d' % i

            if i == 1 and kinks_initializer:
                weights_init, bias_init = get_kinks_initializer(
                    width, params['xmin'], params['xmax'])
            else:
                weights_init = None
                bias_init = tf.zeros_initializer()

            next_out = tf.layers.dense(
                inputs=outputs,
                units=width,
                activation=tf.nn.relu,
                kernel_initializer=weights_init,
                bias_initializer=bias_init,
                name=name)
            i += 1
            outputs = next_out
            save_weight_norm_summaries(outputs, i-1)

        i += 1
        y = tf.layers.dense(
            inputs=outputs,
            units=1,
            activation=None,
            name='output')
        save_weight_norm_summaries(y, i-1)

        pred_dict = {'y': y}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_dict)

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss_name = 'train_loss'
        else:
            info('create model in eval mode')
            loss_name = 'eval_loss'

        y_truth = labels
        loss = tf.reduce_mean(tf.square(y - y_truth), name=loss_name) / 2.

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())

        tf.summary.scalar(loss_name, loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_dict,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=None)

    return model_fn


def create_input_fn_from_data(x, y, epochs=1, batch_size=128, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            dataset = dataset.shuffle(len(x))
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        feature_cols = {'x': features}
        return feature_cols, labels

    return input_fn


def create_prediction_input_fn_from_data(
        x, epochs=1, batch_size=128, shuffle=False):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(x)
        if shuffle:
            dataset = dataset.shuffle(len(x))
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        feature_cols = {'x': features}
        return feature_cols

    return input_fn


def create_input_fn(target_fn, xmin, xmax, n,
                    epochs, batch_size, shuffle=True):
    """Deep regression input pipeline.
    target = target function we are trying to fit.
    n = number of samples to generate.
    p = number of features fed into the model.
    pp = total number of features in the target function.
    """
    x = (xmax - xmin) * np.random.random((n, 1)) + xmin
    # x = x.astype(np.float32)
    y = target_fn(x)
    input_fn = create_input_fn_from_data(x, y, epochs, batch_size, shuffle)
    return input_fn, x, y


def get_base_models_dir():
    """The base directory at which all models are saved."""
    base_dir = os.path.dirname(os.path.realpath(__file__))
    return '%s/models' % base_dir


def get_model_dir(ex_params):
    """Return the directory in which to save model
checkpoints and summaries."""
    widths_str = '-'.join([str(w) for w in ex_params.widths])
    noise_str = 'sigma=%.2f' % ex_params.noise_sigma
    omega_str = 'omega=%.1f' % ex_params.omega
    return '%s/%s-%s-N=%d-%s-%s' % (
        get_base_models_dir(), ex_params.model_name,
        widths_str, ex_params.n_train, noise_str, omega_str
    )


def create_estimator(ex_params):
    """Provide ExperimentParams."""
    config = tf.estimator.RunConfig(
        model_dir=get_model_dir(ex_params),
        save_summary_steps=ex_params.save_summary_steps,
        # also determines when eval happens during training,
        # because checkpoints are needed to run an
        # evaluation.
        save_checkpoints_steps=ex_params.save_checkpoint_steps,
        # how often global step/sec is logged
        log_step_count_steps=10000,
    )

    # Use a custom model_fn
    model_params = {
        'learning_rate': ex_params.learning_rate,
        'widths': ex_params.widths,
        'xmin': ex_params.xmin,
        'xmax': ex_params.xmax,
        }
    model = tf.estimator.Estimator(
        model_fn=ex_params.model_fn,
        params=model_params,
        config=config)

    return model


def steps_per_epoch(n_train, batch_size):
    return (n_train + batch_size - 1) // batch_size


def max_training_steps(n_train, epochs, batch_size):
    max_steps = steps_per_epoch(n_train, batch_size) * epochs
    return max_steps


def train(model, target_fn, ex_params, hooks=[]):
    """Train the model."""
    train_input_fn, train_x, train_y = create_input_fn(
        target_fn, ex_params.xmin, ex_params.xmax,
        ex_params.n_train, ex_params.epochs, ex_params.batch_size)

    logging_hook = tfutils.MyLoggingTensorHook(
        tensors={'train_loss': 'train_loss'},
        every_n_secs=5,
        steps_per_epoch=steps_per_epoch(ex_params.n_train,
                                        ex_params.batch_size))

    hooks.append(logging_hook)

    # Use max_steps so if we restore a trained model we don't
    # continue training it.
    model.train(train_input_fn,
                max_steps=max_training_steps(
                    ex_params.n_train, ex_params.epochs, ex_params.batch_size),
                hooks=hooks)

    return train_x, train_y


class DebugHook(tf.train.SessionRunHook):
    def __init__(self, name):
        self.name = name

    def begin(self):
        info(self.name)

    # def end(self, session):
    #     info('%s: end' % self.name)

    # def before_run(self, run_context):
    #     info(self.name)

    # def after_run(self, run_context, run_values):
    #     info('%s: after run' % self.name)


class KinkLoggerHook(tf.train.SessionRunHook):
    def __init__(self):
        pass

    def before_run(self, run_context):
        layer = 'hidden1'
        tensors = {
            'weights': tfutils._as_graph_element('%s/kernel:0' % layer),
            'biases': tfutils._as_graph_element('%s/bias:0' % layer),
        }
        return SessionRunArgs(tensors)

    def after_run(self, run_context, run_values):
        results = run_values.results
        self.kinks = - results['biases'] / results['weights']
        self.kinks = self.kinks.flatten()


def train_and_eval(model, target_fn, ex_params):
    """Train and evaluate the model."""
    train_input_fn, train_x, train_y = create_input_fn(
        target_fn, ex_params.xmin, ex_params.xmax,
        ex_params.n_train, ex_params.epochs, ex_params.batch_size)
    test_input_fn, test_x, test_y = create_input_fn(
        target_fn, ex_params.xmin, ex_params.xmax,
        ex_params.n_val, epochs=1, batch_size=ex_params.batch_size)

    train_logging_hook = tfutils.MyLoggingTensorHook(
        tensors={'train_loss': 'train_loss'},
        every_n_secs=5,
        steps_per_epoch=steps_per_epoch(ex_params.n_train,
                                        ex_params.batch_size))

    stop_hook = tfutils.StopRequestHook(
        every_n_steps=ex_params.save_eval_plots_steps)

    train_hooks = [train_logging_hook, stop_hook]

    eval_logging_hook = tfutils.MyLoggingTensorHook(
        tensors={'eval_loss': 'eval_loss'},
        every_n_secs=10,
        steps_per_epoch=steps_per_epoch(ex_params.n_train,
                                        ex_params.batch_size))
    kink_hook = KinkLoggerHook()

    eval_hooks = [eval_logging_hook, kink_hook]
    plot_idx = 1
    max_steps = max_training_steps(ex_params.n_train,
                                   ex_params.epochs,
                                   ex_params.batch_size)
    info('max_steps = %d' % max_steps)

    while stop_hook.global_step < max_steps:
        info('========= Training (step=%d) ========='
             % stop_hook.global_step)
        # experiment.train_and_evaluate()
        model.train(train_input_fn, hooks=train_hooks)
        info('========= Evaluating =========')
        model.evaluate(test_input_fn, hooks=eval_hooks)
        # info(eval_result)
        save_eval_plot(model, target_fn, ex_params,
                       stop_hook.global_step, plot_idx,
                       train_x, train_y, test_x, test_y,
                       kinks=kink_hook.kinks)
        plot_idx += 1

    return train_x, train_y, test_x, test_y


def save_eval_plot(model, target_fn, ex_params, step, plot_idx,
                   train_x, train_y, val_x, val_y, kinks=None):
    # train_input_fn = create_input_fn_from_data(train_x, train_y,
    #                                            shuffle=False)
    # train_pred = predict(model, train_input_fn)

    # val_input_fn = create_input_fn_from_data(val_x, val_y, shuffle=False)
    # val_pred = predict(model, val_input_fn)

    num_pred_pts = 1024
    pred_x = np.linspace(ex_params.xmin, ex_params.xmax, num=num_pred_pts)
    pred_x = np.reshape(pred_x, [num_pred_pts, 1])
    pred_input_fn = create_prediction_input_fn_from_data(pred_x)
    pred_y = predict(model, pred_input_fn)

    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    # plt.plot(train_x, train_y, '.', train_x, train_pred, '.')
    plt.xlim(ex_params.xmin, ex_params.xmax)
    plt.plot(train_x, train_y, '+', pred_x, pred_y, '-')

    # Plot vertical lines
    if kinks is not None:
        for k in kinks:
            plt.axvline(k, color='tab:gray', linestyle='--', linewidth=1.0)

    plt.title('Train %s layers=%s N=%d step=%d'
              % (ex_params.model_name,
                 ex_params.widths,
                 ex_params.n_train,
                 step))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(212)
    # plt.plot(val_x, val_y, '.', val_x, val_pred, '.')
    plt.plot(val_x, val_y, '+', pred_x, pred_y, '-')
    plt.title('Validation')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()

    plots_dir = '%s/raw-plots' % get_model_dir(ex_params)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    plt.savefig('%s/plot-%d.png' % (plots_dir, plot_idx))


# def eval(model, target_fn):
#     test_input_fn, test_x, test_y = create_input_fn(
#         target_fn, xmin, xmax, n_val, epochs=1,
#         batch_size=128, shuffle=False)
#     pred = model.predict(test_input_fn)
#     y_pred = [p['y'] for p in pred]
#     test_loss = np.mean(np.square(test_y - y_pred))
#     info('test_loss = %f' % test_loss)

#     test_input_fn, _, _ = create_input_fn(
#         target_fn, xmin, xmax, n_val, epochs=1,
#         batch_size=128, shuffle=False)
#     ev = model.evaluate(input_fn=test_input_fn)
#     info('Evaluation: %s' % str(ev))


def predict(model, input_fn):
    pred = model.predict(input_fn)
    y_pred = [p['y'] for p in pred]
    return np.array(y_pred)


def delete_all_models():
    """Delete all saved models, checkpoints, and events."""
    base_dir = get_base_models_dir()
    if os.path.exists(base_dir):
        info('Deleting whole directory %s/ in a few secs...'
             % base_dir)
        time.sleep(2)
        shutil.rmtree(base_dir)


def delete_model_dir(ex_params):
    """Delete the model directory."""
    model_dir = get_model_dir(ex_params)
    if os.path.exists(model_dir):
        info('Deleting model directory %s/' % model_dir)
        # time.sleep(2)
        shutil.rmtree(model_dir)


def create_experiment_params(args):
    if args['--xmax'] == '2.*pi':
        xmax = 2. * math.pi
    else:
        xmax = float(args['--xmax'])

    model_name = 'regression-relu'

    if args['--kinks-initializer']:
        model_name += '-kinkinit'

    return ExperimentParams(
        model_name=model_name,
        model_fn=create_model_fn(tf.nn.relu, args['--kinks-initializer']),
        widths=[int(w) for w in args['--widths'].split(',')],

        xmin=float(args['--xmin']),
        xmax=xmax,

        noise_sigma=float(args['--noise-sigma']),
        omega=float(args['--omega']),
        learning_rate=float(args['--learning-rate']),

        n_train=int(args['--n-train']),
        n_val=int(args['--n-val']),
        epochs=int(args['--epochs']),
        batch_size=int(args['--batch-size']),
        save_checkpoint_steps=int(args['--eval-every-steps']),
        save_summary_steps=int(args['--save-summary-steps']),
        save_eval_plots_steps=int(args['--save-eval-plots-every-steps']),
    )


def main():
    tf.logging.set_verbosity(tf.logging.WARN)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = docopt(__doc__)
    p = create_experiment_params(args)

    if args['--delete-saved-model']:
        delete_model_dir(p)

    target_fn = create_target_fn(p.noise_sigma, p.omega)
    model = create_estimator(p)

    if args['--train']:
        train_x, train_y = train(model, target_fn, p)
    elif args['--train-and-eval']:
        train_x, train_y, test_x, test_y = train_and_eval(model, target_fn, p)
    else:
        info('Please specify one of --train, --train-and-eval')

    # for idx in range(2):
    #     omega = 2.0**idx
    #     n_train = int(n_train_per_period*omega)

    #     model = create_estimator(model_fn, widths, n_train, learning_rate,
    #                              noise_sigma, model_name, omega)
    #     target_fn = create_target_fn(noise_sigma, omega)
    #     train_and_eval(model, target_fn, n_train, n_val, epochs, batch_size)


if __name__ == '__main__':
    main()
