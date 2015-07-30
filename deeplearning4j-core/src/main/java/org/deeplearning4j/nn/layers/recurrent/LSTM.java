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

package org.deeplearning4j.nn.layers.recurrent;


import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossCalculation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.log;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;
import static org.nd4j.linalg.ops.transforms.Transforms.tanh;

/**
 * LSTM recurrent net.
 *
 * Based on karpathy et. al's
 * work on generation of image descriptions.
 * http://arxiv.org/pdf/1411.4555.pdf
 *https://github.com/karpathy/neuraltalk/blob/master/imagernn/lstm_generator.py
 *
 * @author Adam Gibson
 */
public class LSTM extends BaseLayer {
    //recurrent weights
    private INDArray iFogZ, iFogA, memCellActivations, hIn, outputActivations;
    // update values for drop connect
    private INDArray u, u2;
    //current input // paper has it as image representations
    private INDArray xi;
    //predicted time series // paper has it as word representations
    private INDArray xs;

    public LSTM(NeuralNetConfiguration conf) {
        super(conf);
    }



    /**
     * Forward propagation
     * @param xi the current example
     * @return
     */
    public INDArray forward(INDArray xi) {
        return activate(xi);
    }


    /**
     * Forward propagation
     * @param xi the current example
     * @param xs the time series to predict based on
     * @return
     */
    public INDArray forward(INDArray xi,INDArray xs) {
        this.xs = xs;
        this.xi = xi;
        input = Nd4j.vstack(xi,xs);
        return activate(input);
    }


    /**
     * Back propagation in the given input
     * @param output
     * @return {@link org.deeplearning4j.nn.gradient.Gradient}
     */
    public Gradient backprop(INDArray output) {
        // TODO rework as backpropGradient and determine if call activate here
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHTS);

        //add column of zeros since not used in forward pass
        INDArray dY = Nd4j.vstack(Nd4j.zeros(output.columns()), output);
        INDArray dWd = outputActivations.transpose().mmul(dY);
        INDArray dBd = Nd4j.sum(dY,0);
        INDArray dHout = dY.mmul(decoderWeights.transpose());

        if(conf.isUseDropConnect() & conf.getDropOut() > 0)
            dHout.muli(u2);

        INDArray dIFogZ = Nd4j.zeros(iFogZ.shape());
        INDArray dIFogA = Nd4j.zeros(iFogA.shape());
        INDArray dRecurrentWeights = Nd4j.zeros(recurrentWeights.shape());
        INDArray dHin = Nd4j.zeros(hIn.shape());

        INDArray dC = Nd4j.zeros(memCellActivations.shape());
        INDArray dx = Nd4j.zeros(input.shape());

        int sequenceLen = outputActivations.rows(); // n
        int hiddenLayerSize = outputActivations.columns(); // d


        for(int t = sequenceLen -1; t > 0; t--) {

            if(conf.getActivationFunction().equals("tanh")) {
                INDArray tanhCt = tanh(memCellActivations.slice(t)).mul(dHout.slice(t));
                dIFogA.slice(t).put(new NDArrayIndex[]{interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)}, tanhCt);
                dC.slice(t).addi(pow(tanhCt,2).rsubi(1).muli(iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(dHout.slice(t))));
            }
            else {
                dIFogA.slice(t).put(new NDArrayIndex[]{interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)}, memCellActivations.slice(t).mul(dHout.slice(t)));
                dC.slice(t).addi(iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(dHout.slice(t)));
            }

            if(t > 0) {
                dIFogA.slice(t).put(new NDArrayIndex[]{interval(hiddenLayerSize, 2 * hiddenLayerSize)}, memCellActivations.slice(t - 1).mul(dC.slice(t)));
                dC.slice(t - 1).addi(iFogA.slice(t).get(interval(hiddenLayerSize, 2 * hiddenLayerSize)).mul(dC.slice(t)));
            }

            dIFogA.slice(t).put(new NDArrayIndex[]{interval(0, hiddenLayerSize)}, iFogA.slice(t).get(interval(3 * hiddenLayerSize, iFogA.columns())).mul(dC.slice(t)));
            dIFogA.slice(t).put(new NDArrayIndex[]{interval(3 * hiddenLayerSize, dIFogA.columns())}, iFogA.slice(t).get(interval(0,hiddenLayerSize)).mul(dC.slice(t)));

            dIFogZ.slice(t).put(new NDArrayIndex[]{interval(3 * hiddenLayerSize, dIFogZ.columns())},pow(iFogA.slice(t).get(interval(3 * hiddenLayerSize, iFogA.columns())),2).rsubi(1).mul(dIFogA.slice(t).get(interval(3 * hiddenLayerSize, dIFogA.columns()))));
            INDArray prediction = iFogA.slice(t).get(interval(0,3 * hiddenLayerSize));
            dIFogZ.slice(t).put(new NDArrayIndex[]{interval(0, 3 * hiddenLayerSize)}, prediction.mul(prediction.rsub(1)).mul(dIFogA.slice(t).get(interval(0, 3 * hiddenLayerSize))));

            dRecurrentWeights.addi(hIn.slice(t).transpose().mmul(dIFogZ.slice(t)));
            dHin.slice(t).assign(dIFogZ.slice(t).mmul(recurrentWeights.transpose()));

            dx.slice(t).assign(dHin.slice(t).get(interval(1, 1 + hiddenLayerSize)));
            if(t > 0)
                dHout.slice(t - 1).addi(dHin.slice(t).get(interval(1 + hiddenLayerSize, dHin.columns())));

            if(conf.isUseDropConnect() & conf.getDropOut() > 0)
                dx.muli(u);

        }

        //TODO still use this?
        clear();

        Gradient gradient = new DefaultGradient();
        gradient.gradientForVariable().put(LSTMParamInitializer.DECODER_BIAS,dBd);
        gradient.gradientForVariable().put(LSTMParamInitializer.DECODER_WEIGHTS,dWd);
        gradient.gradientForVariable().put(LSTMParamInitializer.RECURRENT_WEIGHTS,dRecurrentWeights);
        return gradient;
    }


    @Override
    public INDArray input() {
        // TODO include xs?
        return xi;
    }

    @Override
    public INDArray activate(INDArray input, boolean training){
        setInput(input, training);
        INDArray prevOutputActivations, prevMemCellActivations;

        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHTS); // WLSTM
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray bias = getParam(LSTMParamInitializer.DECODER_BIAS);

        if(conf.isUseDropConnect() && training) {
            if (conf.getDropOut() > 0) {
                // TODO verfiy do we go with our applyDropconnect or use the one from Karpathy?
//            double scale = 1 / (1 - conf.getDropOut());
//            u = Nd4j.rand(input.shape()).lti(1 - conf.getDropOut()).muli(scale);
//            input.muli(u);
                u = Dropout.applyDropout(input, conf.getDropOut(), u);
                input.muli(u);
            }
        }

        int inputSequenceLen = input.rows(); // n - not miniBatch
        int hiddenLayerSize = decoderWeights.rows(); // hidden layer size
        int recurrentSize = recurrentWeights.size(0);

        //Allocate arrays for activations:
        //xt, ht-1, bias - input activations
        hIn = Nd4j.zeros(inputSequenceLen, recurrentSize);
        //hOUt hidden layer output activations
        outputActivations = Nd4j.zeros(inputSequenceLen, hiddenLayerSize);
        // C memCellActivations
        memCellActivations = Nd4j.zeros(inputSequenceLen, hiddenLayerSize);

        // iFog linear transformation w/ no bias
        iFogZ = Nd4j.zeros(inputSequenceLen, hiddenLayerSize * 4);
        // iFogF activations
        iFogA = Nd4j.zeros(inputSequenceLen, hiddenLayerSize * 4);


        for(int t = 0; t < inputSequenceLen ; t++) {
            prevOutputActivations = t == 0 ? Nd4j.zeros(hiddenLayerSize) : outputActivations.getRow(t - 1);
            prevMemCellActivations = t == 0 ? Nd4j.zeros(hiddenLayerSize) : memCellActivations.getRow(t - 1);

            hIn.put(t, 0, 1.0);
            hIn.slice(t).put(new NDArrayIndex[]{interval(1, 1 + hiddenLayerSize)}, input.slice(t));
            hIn.slice(t).put(new NDArrayIndex[]{interval(1 + hiddenLayerSize, 2 * hiddenLayerSize + 1)}, prevOutputActivations);

            //compute all gate linear transformations
            iFogZ.putRow(t, hIn.slice(t).mmul(recurrentWeights));

            //store activations for i, f, o
            iFogA.slice(t).put(new NDArrayIndex[]{interval(0, 3 * hiddenLayerSize)}, sigmoid(iFogZ.slice(t).get(new NDArrayIndex[]{interval(0, 3 * hiddenLayerSize)})));

            // store activations for c
            iFogA.slice(t).put(new NDArrayIndex[]{interval(3 * hiddenLayerSize, iFogA.columns() - 1)},
                    tanh(iFogZ.slice(t).get(interval(3 * hiddenLayerSize, iFogZ.columns() - 1))));

            //i dot product h(WcxXt + WcmMt-1)
            memCellActivations.putRow(t, iFogA.slice(t).get(interval(0, hiddenLayerSize)).mul(iFogA.slice(t).get(interval(3 * hiddenLayerSize, iFogA.columns()))));

            if(t > 0)
                // Ct curr memory cell activations after t 0
                memCellActivations.slice(t).addi(iFogA.slice(t).get(interval(hiddenLayerSize, 2 * hiddenLayerSize)).mul(prevMemCellActivations));

            // mt hidden out or output before activation
            if(conf.getActivationFunction().equals("tanh")) {
                outputActivations.slice(t).assign(iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(tanh(memCellActivations.getRow(t))));
            } else {
                outputActivations.slice(t).assign(iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(memCellActivations.getRow(t)));
            }
        }

        if(conf.isUseDropConnect() && training) {
            if (conf.getDropOut() > 0) {
                u2 = Dropout.applyDropout(outputActivations, conf.getDropOut(), u2);
                outputActivations.muli(u2);
            }
        }
        return outputActivations.get(interval(1, outputActivations.rows())).mmul(decoderWeights).addiRowVector(bias);

    }

    /**
     * Prediction with beam search
     * @param xi
     * @param ws
     * @return
     */
    public Collection<Pair<List<Integer>,Double>> predict(INDArray xi,INDArray ws) {
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        int d = decoderWeights.rows();
        Triple<INDArray,INDArray,INDArray> yhc = lstmTick(xi, Nd4j.zeros(d), Nd4j.zeros(d));
        BeamSearch search = new BeamSearch(20,ws,yhc.getSecond(),yhc.getThird());
        return search.search();
    }


    @Override
    public  void clear() {
        u = null;
        hIn = null;
        outputActivations = null;
        iFogZ = null;
        iFogA = null;
        memCellActivations = null;
        input = null;
        u2 = null;
    }


    private  class BeamSearch {
        private List<Beam> beams = new ArrayList<>();
        private int nSteps = 0;
        private INDArray h,c;
        private INDArray ws;
        private int beamSize = 5;
        public BeamSearch(int nSteps,INDArray ws, INDArray h, INDArray c) {
            this.nSteps = nSteps;
            this.h = h;
            this.c = c;
            this.ws = ws;
            beams.add(new Beam(0.0,new ArrayList<Integer>(),h,c));

        }

        public Collection<Pair<List<Integer>,Double>> search() {
            if(beamSize > 1) {
                while(true) {
                    List<Beam> candidates = new ArrayList<>();
                    for(Beam beam : beams) {
                        //  ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
                        int ixPrev = beam.getIndices().get(beam.getIndices().size() - 1);
                        if(ixPrev == 0 && !beam.getIndices().isEmpty()) {
                            candidates.add(beam);
                            continue;
                        }

                        Triple<INDArray,INDArray,INDArray> yhc = lstmTick(ws.slice(ixPrev),beam.getHidden(),beam.getC());
                        INDArray y1 = yhc.getFirst().ravel();
                        double maxy1 = y1.max(Integer.MAX_VALUE).getDouble(0);
                        INDArray e1 = exp(y1.subi(maxy1));
                        INDArray p1 = e1.divi(Nd4j.sum(e1,Integer.MAX_VALUE));
                        y1 = log(p1.addi(Nd4j.EPS_THRESHOLD));
                        //indices/sorted arrays
                        INDArray[] topIndices = Nd4j.sortWithIndices(y1,0,false);
                        for(int i = 0; i < beamSize; i++) {
                            int idx = topIndices[0].getInt(i);
                            List<Integer> beamCopy = new ArrayList<>(beam.getIndices());
                            beamCopy.add(idx);
                            candidates.add(new Beam(beam.getLogProba() + y1.getDouble(idx),beamCopy,yhc.getSecond(),yhc.getThird()));
                        }


                    }

                    //sort the beams
                    //truncate beams to be of beam size also setting beams = candidates
                    nSteps++;
                    if(nSteps >= 20)
                        break;

                }

                List<Pair<List<Integer>,Double>> ret = new ArrayList<>();
                for(Beam b : beams) {
                    ret.add(new Pair<>(b.getIndices(),b.getLogProba()));
                }

                return ret;


            }

            else {
                int ixPrev = 0;
                double predictedLogProba = 0.0;
                List<Integer> predix = new ArrayList<>();
                while(true) {
                    Triple<INDArray,INDArray,INDArray> yhc = lstmTick(ws.slice(ixPrev),h,c);
                    Pair<Integer,Double> yMax = yMax(yhc.getFirst());
                    predix.add(yMax.getFirst());
                    predictedLogProba += yMax.getSecond();

                    nSteps++;
                    if(ixPrev == 0 || nSteps >= 20)
                        break;
                }

                return Collections.singletonList(new Pair<>(predix,predictedLogProba));

            }

        }
    }

    private Pair<Integer,Double> yMax(INDArray y) {
        INDArray y1 = y.linearView();
        double max = y.max(Integer.MAX_VALUE).getDouble(0);
        INDArray e1 = exp(y1.rsub(max));
        INDArray p1 = e1.divi(e1.sum(Integer.MAX_VALUE));
        y1 = log(p1.addi(Nd4j.EPS_THRESHOLD));
        INDArray[] sorted = Nd4j.sortWithIndices(y1,0,true);
        int ix = sorted[0].getInt(0);
        return new Pair<>(ix,sorted[1].getDouble(ix));
    }

    private static class Beam {
        private double logProba = 0.0;
        private List<Integer> indices;
        //hidden and cell states
        private INDArray hidden,c;

        public Beam(double logProba, List<Integer> indices, INDArray hidden, INDArray c) {
            this.logProba = logProba;
            this.indices = indices;
            this.hidden = hidden;
            this.c = c;
        }

        public double getLogProba() {
            return logProba;
        }

        public void setLogProba(double logProba) {
            this.logProba = logProba;
        }

        public List<Integer> getIndices() {
            return indices;
        }

        public void setIndices(List<Integer> indices) {
            this.indices = indices;
        }

        public INDArray getHidden() {
            return hidden;
        }

        public void setHidden(INDArray hidden) {
            this.hidden = hidden;
        }

        public INDArray getC() {
            return c;
        }

        public void setC(INDArray c) {
            this.c = c;
        }
    }

    private Triple<INDArray,INDArray,INDArray> lstmTick(INDArray x,INDArray hPrev,INDArray cPrev) {
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHTS);
        INDArray decoderBias = getParam(LSTMParamInitializer.DECODER_BIAS);

        int t = 0;
        int d = decoderWeights.rows();
        INDArray hIn = Nd4j.zeros(1,recurrentWeights.rows());
        hIn.putRow(0,Nd4j.ones(hIn.columns()));
        hIn.slice(t).put(new NDArrayIndex[]{interval(1,1 + d)},x);
        hIn.slice(t).put(new NDArrayIndex[]{interval(1 + d,hIn.columns())},hPrev);


        INDArray iFog = Nd4j.zeros(1, d * 4);
        INDArray iFogf = Nd4j.zeros(iFog.shape());
        INDArray c = Nd4j.zeros(d);
        iFog.putScalar(t,hIn.slice(t).mmul(recurrentWeights).getDouble(0));
        NDArrayIndex[] indices = new NDArrayIndex[]{interval(0,3 * d)};
        iFogf.slice(t).put(indices,sigmoid(iFogA.slice(t).get(indices)));
        NDArrayIndex[] after = new NDArrayIndex[]{interval(3 * d,iFogf.columns())};
        iFogf.slice(t).put(after,tanh(iFogf.slice(t).get(after)));
        c.slice(t).assign(iFogf.slice(t).get(interval(0,d)).mul(iFogf.slice(t).get(interval(3 * d,iFogf.columns()))).addi(iFogf.slice(t).get(interval(d, 2 * d))).muli(cPrev));

        if(conf.getActivationFunction().equals("tanh"))
            outputActivations.slice(t).assign(iFogf.slice(t).get(interval(2 * d,3 * d)).mul(tanh(c.slice(t))));
        else
            outputActivations.slice(t).assign(iFogf.slice(t).get(interval(2 * d,3 * d)).mul(c.slice(t)));
        INDArray y = outputActivations.mmul(decoderWeights).addiRowVector(decoderBias);
        return new Triple<>(y, outputActivations, c);


    }


    @Override
    public void fit() {
        Solver solver = new Solver.Builder()
                .model(this).configure(conf()).listeners(getListeners())
                .build();
        solver.optimize();
    }



    @Override
    public void update(INDArray gradient, String paramType) {
        setParams(params().subi(gradient));
        computeGradientAndScore();

    }


    @Override
    public double l2Magnitude() {
        return Transforms.pow(getParam(LSTMParamInitializer.RECURRENT_WEIGHTS), 2).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public double l1Magnitude() {
        return Transforms.abs(getParam(LSTMParamInitializer.RECURRENT_WEIGHTS)).sum(Integer.MAX_VALUE).getDouble(0);
    }


    @Override
    public void computeGradientAndScore() {
        INDArray forward = forward(xi, xs);
        INDArray probas = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax",forward).derivative(),1);
        gradient = backprop(probas);
        if (conf.getLossFunction() == LossFunctions.LossFunction.CUSTOM) {
            LossFunction create = Nd4j.getOpFactory().createLossFunction(conf.getCustomLossFunction(), input, forward);
            create.exec();
            score = create.currentResult().doubleValue();
        }

        else {
            score = LossCalculation.builder()
                    .l1(conf.getL1()).l2(conf.getL2())
                    .l1Magnitude(l1Magnitude()).l2Magnitude(l2Magnitude())
                    .labels(xs).z(probas).lossFunction(conf.getLossFunction())
                    .useRegularization(conf.isUseRegularization()).build().score();

        }
    }


    @Override
    public INDArray transform(INDArray data) {
        return  Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", activate(data)).derivative(), 1);
    }



    @Override
    public void setParams(INDArray params) {
        int count = 0;
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHTS);
        INDArray decoderBias = getParam(LSTMParamInitializer.DECODER_BIAS);

        INDArray recurrentWeightsLinear = recurrentWeights.linearView();
        INDArray decoderWeightsLinear = decoderWeights.linearView();
        INDArray decoderBiasLinear = decoderBias.linearView();
        int recurrentPlusDecoder = recurrentWeightsLinear.length() + decoderWeightsLinear.length();
        boolean pastRecurrentWeights = false;
        for(int i = 0; i < params.length(); i++) {
            //reset once for normal recurrent weights
            if(count == recurrentWeightsLinear.length()) {
                count = 0;
                pastRecurrentWeights = true;
            }
            //reset again for decoder weights, no need to do this as this sets up the bias count properly
            else if(count == decoderWeightsLinear.length() && pastRecurrentWeights)
                count = 0;

            if(i < recurrentWeights.length())
                recurrentWeights.linearView().putScalar(count++,params.getDouble(i));

            else if(i < recurrentPlusDecoder)
                decoderWeightsLinear.putScalar(count++,params.getDouble(i));
            else
                decoderBiasLinear.putScalar(count++,params.getDouble(i));

        }
    }

    @Override
    public void fit(INDArray data) {
        xi = data.slice(0);
        NDArrayIndex[] everythingElse = {
                NDArrayIndex.interval(1,data.rows()),NDArrayIndex.interval(0,data.columns())
        };
        xs = data.get(everythingElse);
        Solver solver = new Solver.Builder()
                .configure(conf).model(this).listeners(getListeners())
                .build();
        solver.optimize();
    }

    @Override
    public void iterate(INDArray input) {

    }



    @Override
    public int batchSize() {
        return xi.rows();
    }

}
