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

package org.deeplearning4j.nn.conf;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Output post processor's handle layer to layer interactions
 * to ensure things like proper shape for input among other things.
 *
 * @author Adam Gibson
 */
@Deprecated
public interface OutputPostProcessor extends Serializable {
    /**
     * Used for handling pre processing of layer output.
     * The typical use case is for handling reshaping of output
     * in to shapes proper for the next layer of input.
     * @param output the layer output to post preProcess
     * @return the processed output
     */
    INDArray process(INDArray output);


    /**Reverse the preProcess for backprop. Process Gradient/epsilon
     * (from layer above) before using them to calculate the backprop
     * gradient for this layer.
     * @param input the reverse preProcess
     * @return the reverse of the pre preprocessing on the output
     */
    INDArray backprop(INDArray input);

}
