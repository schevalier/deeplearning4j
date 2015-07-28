/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.classifiers.lstm;

import java.util.Arrays;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;


/**
 * Based on GravesLSTMTest by Alex Black
 */
public class LSTMTest {

    private static final Logger log = LoggerFactory.getLogger(LSTMTest.class);

    @Test
    public void testTraffic() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("tanh")
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .layer(new org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                        .nIn(4)
                        .nOut(4)
                        .build())
                .build();

        LSTM layer = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(10)),0);

        INDArray predict = FeatureUtil.toOutcomeMatrix(new int[]{0,1,2,3},4);
        INDArray out = layer.activate(predict);
        log.info("Out " + out);
    }


    @Test
    public void testLSTMForwardBasic() {
        int nIn = 13;
        int nHiddenUnits = 17;


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("tanh")
                .layer(new org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                        .nIn(nIn)
                        .nOut(nHiddenUnits)
                        .build())
                .build();

        LSTM layer = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(10)),0);

        INDArray dataSingleExample = Nd4j.ones(1, nIn);
        INDArray activations1 = layer.activate(dataSingleExample);
        assertArrayEquals(activations1.shape(),new int[]{1,nHiddenUnits});

        INDArray dataMultiExample = Nd4j.ones(10,nIn);
        INDArray activations2 = layer.activate(dataMultiExample);
        assertArrayEquals(activations2.shape(),new int[]{10,nHiddenUnits});

    }

    @Test
    public void testLSTMBackwardBasic(){
        //Very basic test of backprop for mini-batch + time series
        //Essentially make sure it doesn't throw any exceptions, and provides output in the correct shape.

        testLSTMBackwardBasicHelper(13, 3, 17, 10);
        testLSTMBackwardBasicHelper(13, 3, 17, 1);		//Edge case: miniBatchSize = 1
    }

    private static void testLSTMBackwardBasicHelper(int nIn, int nOut, int lstmNHiddenUnits, int miniBatchSize ){

        INDArray inputData = Nd4j.ones(miniBatchSize,nIn);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("tanh")
                .dist(new UniformDistribution(0, 1))
                .layer(new org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                        .nIn(nIn)
                        .nOut(lstmNHiddenUnits)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .build())
                .build();

        LSTM lstm = LayerFactories.getFactory(conf.getLayer()).create(conf);
        //Set input, do a forward pass:
        lstm.activate(inputData);
        assertNotNull(lstm.input());

        //Create pseudo-gradient for input to LSTM layer (i.e., as if created by OutputLayer)
        //This should have two elements: bias and weight gradients.
        Gradient gradient = createPrevGradient(miniBatchSize, nOut, lstmNHiddenUnits);
        Layer prevLayer = createOutputLayer();

        INDArray epsilon = Nd4j.ones(miniBatchSize, lstmNHiddenUnits);

        Pair<Gradient,INDArray> out = lstm.backpropGradient(epsilon, gradient, prevLayer);
        Gradient outGradient = out.getFirst();
        INDArray nextEpsilon = out.getSecond();

        INDArray biasGradient = outGradient.getGradientFor(LSTMParamInitializer.DECODER_BIAS);
        INDArray inWeightGradient = outGradient.getGradientFor(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray recurrentWeightGradient = outGradient.getGradientFor(LSTMParamInitializer.RECURRENT_WEIGHTS);
        assertNotNull(biasGradient);
        assertNotNull(inWeightGradient);
        assertNotNull(recurrentWeightGradient);

        assertArrayEquals(biasGradient.shape(),new int[]{1,4*lstmNHiddenUnits});
        assertArrayEquals(inWeightGradient.shape(),new int[]{nIn,4*lstmNHiddenUnits});
        assertArrayEquals(recurrentWeightGradient.shape(),new int[]{lstmNHiddenUnits,4*lstmNHiddenUnits+3});

        assertNotNull(nextEpsilon);
        assertArrayEquals(nextEpsilon.shape(),new int[]{miniBatchSize,nIn});

        //Check update:
        for( String s : outGradient.gradientForVariable().keySet() ){
            lstm.update(outGradient.getGradientFor(s), s);
        }
    }

    private static Gradient createPrevGradient(int miniBatchSize, int nOut, int lstmNHiddenUnits) {
        Gradient gradient = new DefaultGradient();
        INDArray pseudoBiasGradients = Nd4j.ones(miniBatchSize,nOut);
        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY,pseudoBiasGradients);
        INDArray pseudoWeightGradients = Nd4j.ones(miniBatchSize,lstmNHiddenUnits,nOut);
        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, pseudoWeightGradients);
        return gradient;
    }

    private static Layer createOutputLayer() {
        NeuralNetConfiguration outputConf = new NeuralNetConfiguration.Builder()
                .activationFunction("softmax")
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nIn(3)
                        .nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .build())
                .build();

        return LayerFactories.getFactory(outputConf.getLayer()).create(outputConf);
    }


}
