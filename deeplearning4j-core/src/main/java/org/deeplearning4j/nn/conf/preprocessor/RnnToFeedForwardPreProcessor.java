package org.deeplearning4j.nn.conf.preprocessor;

import lombok.EqualsAndHashCode;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**A preprocessor to allow RNN and feed-forward network layers to be used together.<br>
 * For example, GravesLSTM -> OutputLayer or GravesLSTM -> DenseLayer<br>
 * This does two things:<br>
 * (a) Reshapes activations out of RNN layer (which is 3D with shape 
 * [miniBatchSize,layerSize,timeSeriesLength]) into 2d activations (with shape
 * [miniBatchSize*timeSeriesLength,layerSize]) suitable for use in feed-forward layers.<br>
 * (b) Reshapes 2d epsilons (weights*deltas from feed forward layer, with shape
 * [miniBatchSize*timeSeriesLength,layerSize]) into 3d epsilons (with shape
 * [miniBatchSize,layerSize,timeSeriesLength]) for use in RNN layer
 * @author Alex Black
 * @see FeedForwardToRnnPreProcessor for opposite case (i.e., DenseLayer -> GravesLSTM etc)
 */
@EqualsAndHashCode
public class RnnToFeedForwardPreProcessor implements InputPreProcessor {
	private static final long serialVersionUID = -2334789406636365730L;
	private final int timeSeriesLength;
	
	/**@param timeSeriesLength Length of time series training data
	 */
	public RnnToFeedForwardPreProcessor(int timeSeriesLength ) {
		this.timeSeriesLength = timeSeriesLength;
	}
	
	@Override
	public INDArray preProcess(INDArray input) {
		//Need to reshape RNN activations (3d) activations to 2d (for input into feed forward layer)
		if( input.rank() != 3 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 3 (i.e., activations for RNN layer)");
		
		int[] shape = input.shape();
		INDArray permuted = input.permute(0,2,1);	//Permute, so we get correct order after reshaping
		return permuted.reshape(shape[0]*timeSeriesLength,shape[1]);
	}

	@Override
	public INDArray backprop(INDArray output) {
		//Need to reshape FeedForward layer epsilons (2d) to 3d (for use in RNN layer backprop calculations)
		if( output.rank() != 2 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2 (i.e., epsilons from feed forward layer)");
		
		int[] shape = output.shape();
		INDArray reshaped = output.reshape(shape[0]/timeSeriesLength,timeSeriesLength,shape[1]);
		return reshaped.permute(0,2,1);
	}
}
