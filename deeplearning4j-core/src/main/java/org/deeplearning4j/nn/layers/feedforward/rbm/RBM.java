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

package org.deeplearning4j.nn.layers.feedforward.rbm;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.util.RBMUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;
import static org.nd4j.linalg.ops.transforms.Transforms.max;


/**
 * Restricted Boltzmann Machine.
 *
 * Markov chain with gibbs sampling.
 *
 * Supports the following visible units:
 *
 *     binary
 *     gaussian
 *     softmax
 *     linear
 *
 * Supports the following hidden units:
 *     rectified
 *     binary
 *     gaussian
 *     softmax
 *     linear
 *
 * Based on Hinton et al.'s work
 *
 * Great reference:
 * http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239
 *
 *
 * @author Adam Gibson
 *
 */
public  class RBM extends BasePretrainNetwork {

    private transient Random rng;

    public RBM(NeuralNetConfiguration conf) {
        super(conf);
        this.rng = Nd4j.getRandom();
    }

    public RBM(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        this.rng = Nd4j.getRandom();
    }

    /**
     *
     */
    //variance matrices for gaussian visible/hidden units
    protected INDArray sigma,hiddenSigma;


    /**
     * Contrastive divergence revolves around the idea
     * of approximating the log likelihood around x1(input) with repeated sampling.
     * Given is an energy based model: the higher k is (the more we sample the model)
     * the more we lower the energy (increase the likelihood of the model)
     *
     * and lower the likelihood (increase the energy) of the hidden samples.
     *
     * Other insights:
     *    CD - k involves keeping the first k samples of a gibbs sampling of the model.

     */
    public void contrastiveDivergence() {
        Gradient gradient = gradient();
        getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY).subi(gradient.gradientForVariable().get(PretrainParamInitializer.VISIBLE_BIAS_KEY));
        getParam(PretrainParamInitializer.BIAS_KEY).subi(gradient.gradientForVariable().get(PretrainParamInitializer.BIAS_KEY));
        getParam(PretrainParamInitializer.WEIGHT_KEY).subi(gradient.gradientForVariable().get(PretrainParamInitializer.WEIGHT_KEY));
    }


    @Override
    public void computeGradientAndScore() {
        int k = conf.getK();

        //POSITIVE PHASE
        Pair<INDArray,INDArray> probHidden = sampleHiddenGivenVisible(input());

		/*
		 * Start the gibbs sampling.
		 */
        INDArray chainStart = probHidden.getSecond();

		/*
		 * Note that at a later date, we can explore alternative methods of
		 * storing the chain transitions for different kinds of sampling
		 * and exploring the search space.
		 */
        Pair<Pair<INDArray,INDArray>,Pair<INDArray,INDArray>> matrices;
        //negative visible means or expected values
        INDArray nvMeans = null;
        //negative value samples
        INDArray nvSamples = null;
        //negative hidden means or expected values
        INDArray nhMeans = null;
        //negative hidden samples
        INDArray nhSamples = null;

		/*
		 * K steps of gibbs sampling. This is the positive phase of contrastive divergence.
		 *
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values
		 * or averages.
		 *
		 */

        for(int i = 0; i < k; i++) {

            //NEGATIVE PHASE
            if(i == 0)
                matrices = gibbhVh(chainStart);
            else
                matrices = gibbhVh(nhSamples);

            //get the cost updates for sampling in the chain after k iterations
            nvMeans = matrices.getFirst().getFirst();
            nvSamples = matrices.getFirst().getSecond();
            nhMeans = matrices.getSecond().getFirst();
            nhSamples = matrices.getSecond().getSecond();
        }

		/*
		 * Update gradient parameters
		 */
        INDArray wGradient = input().transposei().mmul(probHidden.getSecond()).subi(
                nvSamples.transposei().mmul(nhMeans)
        );



        INDArray hBiasGradient;

        if(conf.getSparsity() != 0)
            //all hidden units must stay around this number
            hBiasGradient = probHidden.getSecond().rsub(conf.getSparsity()).sum(0);
        else
            //update rule: the expected values of the hidden input - the negative hidden  means adjusted by the learning rate
            hBiasGradient = probHidden.getSecond().sub(nhMeans).sum(0);

        //update rule: the expected values of the input - the negative samples adjusted by the learning rate
        INDArray  delta = input.sub(nvSamples);
        INDArray  vBiasGradient =delta.sum(0);

        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(PretrainParamInitializer.VISIBLE_BIAS_KEY,vBiasGradient);
        ret.gradientForVariable().put(PretrainParamInitializer.BIAS_KEY,hBiasGradient);
        ret.gradientForVariable().put(PretrainParamInitializer.WEIGHT_KEY,wGradient);
        gradient = ret;
        setScoreWithZ(delta);
    }





    @Override
    public Layer transpose() {
        RBM r = (RBM) super.transpose();
        org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit h = RBMUtil.inverse(conf.getVisibleUnit());
        org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit v = RBMUtil.inverse(conf.getHiddenUnit());
        if(h == null)
            h = conf.getHiddenUnit();
        if(v == null)
            v = conf.getVisibleUnit();

        r.sigma = sigma;
        r.hiddenSigma = hiddenSigma;
        return r;
    }


    /**
     * Binomial sampling of the hidden values given visible
     * @param v the visible values
     * @return a binomial distribution containing the expected values and the samples
     */
    @Override
    public  Pair<INDArray,INDArray> sampleHiddenGivenVisible(INDArray v) {
        INDArray h1Mean = propUp(v);
        INDArray h1Sample;

        switch (conf.getHiddenUnit()) {
            case RECTIFIED: {
                INDArray sigH1Mean = sigmoid(h1Mean);
		/*
		 * Rectified linear part
		 */
                INDArray sqrtSigH1Mean = sqrt(sigH1Mean);
                INDArray sample = Nd4j.getDistributions().createNormal(h1Mean, 1).sample(h1Mean.shape());
                sample.muli(sqrtSigH1Mean);
                h1Sample = h1Mean.add(sample);
                h1Sample = max(h1Sample, 0.0);
                break;
            }
            case GAUSSIAN: {
                h1Sample = h1Mean.add(Nd4j.randn(h1Mean.rows(), h1Mean.columns(), rng));
                break;
            }
            case SOFTMAX: {
                h1Sample = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", h1Mean), 0);
                break;
            }
            case BINARY: {
                h1Sample = Nd4j.getDistributions().createBinomial(1, h1Mean).sample(h1Mean.shape());
                break;
            }
            default:
                throw new IllegalStateException("Hidden unit type must either be rectified linear or binary");
        }

        return new Pair<>(h1Mean, h1Sample);
    }

    /**
     * Gibbs sampling step: hidden ---> visible ---> hidden
     * @param h the hidden input
     * @return the expected values and samples of both the visible samples given the hidden
     * and the new hidden input and expected values
     */
    public Pair<Pair<INDArray,INDArray>,Pair<INDArray,INDArray>> gibbhVh(INDArray h) {
        Pair<INDArray,INDArray> v1MeanAndSample = sampleVisibleGivenHidden(h);
        INDArray vSample = v1MeanAndSample.getSecond();

        Pair<INDArray,INDArray> h1MeanAndSample = sampleHiddenGivenVisible(vSample);
        return new Pair<>(v1MeanAndSample,h1MeanAndSample);
    }


    /**
     * Guess the visible values given the hidden
     * @param h the hidden units
     * @return a visible mean and sample relative to the hidden states
     * passed in
     */
    @Override
    public Pair<INDArray,INDArray> sampleVisibleGivenHidden(INDArray h) {
        INDArray v1Mean = propDown(h);
        INDArray v1Sample;

        switch (conf.getVisibleUnit()) {
            case GAUSSIAN: {
                v1Sample = v1Mean.add(Nd4j.randn(v1Mean.rows(), v1Mean.columns(), rng));
                break;
            }
            case LINEAR: {
                v1Sample = Nd4j.getDistributions().createNormal(v1Mean, 1).sample(v1Mean.shape());
                break;
            }
            case SOFTMAX: {
                v1Sample = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", v1Mean), 0);
                break;
            }
            case BINARY: {
                v1Sample = Nd4j.getDistributions().createBinomial(1, v1Mean).sample(v1Mean.shape());
                break;
            }
            default: {
                throw new IllegalStateException("Visible type must be one of Binary, Gaussian, SoftMax or Linear");
            }
        }

        return new Pair<>(v1Mean, v1Sample);

    }

    /**
     * Calculates the activation of the visible :
     * sigmoid(v * W + hbias)
     * @param v the visible layer
     * @return the approximated activations of the visible layer
     */
    public INDArray propUp(INDArray v) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        if(conf.isUseDropConnect()) {
            if (conf.getDropOut() > 0) {
                W = W.mul(Nd4j.getDistributions().createBinomial(1,conf.getDropOut()).sample(W.shape()).divi(conf.getDropOut()));
            }
        }
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);

        if(conf.getVisibleUnit() == org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN)
            this.sigma = v.var(0).divi(input.rows());

        INDArray preSig = v.mmul(W).addiRowVector(hBias);

        switch (conf.getHiddenUnit()) {
            case RECTIFIED:
                preSig = max(preSig, 0.0);
                return preSig;
            case GAUSSIAN:
                preSig.addi(Nd4j.randn(preSig.rows(), preSig.columns(), rng));
                return preSig;
            case BINARY:
                return sigmoid(preSig);
            case SOFTMAX:
                return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", preSig), 0);
            default:
                throw new IllegalStateException("Hidden unit type should either be binary, gaussian, or rectified linear");
        }

    }

    /**
     * Calculates the activation of the hidden:
     * activation(h * W + vbias)
     * @param h the hidden layer
     * @return the approximated output of the hidden layer
     */
    public INDArray propDown(INDArray h) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY).transpose();
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);

        INDArray vMean = h.mmul(W).addiRowVector(vBias);

        switch (conf.getVisibleUnit()) {
            case GAUSSIAN:
                INDArray sample = Nd4j.getDistributions().createNormal(vMean, 1).sample(vMean.shape());
                vMean.addi(sample);
                return vMean;
            case LINEAR:
                return vMean;
            case BINARY:
                return sigmoid(vMean);
            case SOFTMAX:
                return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", vMean), 0);
            default:
                throw new IllegalStateException("Visible unit type should either be binary or gaussian");
        }

    }


    /**
     * Reconstructs the visible INPUT.
     * A reconstruction is a propdown of the reconstructed hidden input.
     * @param  training true or false
     * @return the reconstruction of the visible input
     */
    @Override
    public INDArray activate(boolean training) {
        //reconstructed: propUp ----> hidden propDown to transform
        INDArray propUp = propUp(input);
        return propUp;
    }

    /**
     * Note: k is the first input hidden params.
     */
    @Override
    public void fit(INDArray input) {
        if(conf.getVisibleUnit() == org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN) {
            this.sigma = input.var(0);
            this.sigma.divi(input.rows());
        }

        super.fit(input);
    }



    @Override
    public void iterate(INDArray input) {
        if(conf.getVisibleUnit() == org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN)
            this.sigma = input.var(0).divi(input.rows());

        this.input = input.dup();
        applyDropOutIfNecessary(this.input,true);
        contrastiveDivergence();
    }




}
