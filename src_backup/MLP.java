import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;


public class MLP {
	public final int n;
	public final int p;
	public final double alpha;
	private SimpleMatrix wHid;
	private SimpleMatrix wOut;
	private double[] means;
	private double[] stdevs;
	
	public MLP(int n, int p, double alpha){
		this.n = n;
		this.p = p;
		this.alpha = alpha;
	}
	
	public void reset(){
		Random rand = new Random();
		wHid = SimpleMatrix.random(p, n+1, -1, 1, rand);
		wOut = SimpleMatrix.random(n, p+1, -1, 1, rand);
	}
	
	public double[] train(List<double[]> dataSetOrig, int epNum){
		reset();
		
		// Normalization
		List<double[]> dataSet = new ArrayList<double[]>(dataSetOrig.size());
		means = new double[n];
		for(double[] data : dataSetOrig){
			for(int i = 0; i < n; i++){ means[i] += data[i]; }
		}
		for(int i = 0; i < n; i++){ means[i] /= dataSetOrig.size(); }
		
		stdevs = new double[n];
		for(double[] data : dataSetOrig){
			for(int i = 0; i < n; i++){
				double deMean = data[i] - means[i];
				stdevs[i] += deMean * deMean;
			}
		}
		for(int i = 0; i < n; i++){ stdevs[i] = Math.sqrt(stdevs[i] / dataSetOrig.size()); }
		
		for(double[] data : dataSetOrig){
			double[] normData = new double[n];
			for(int i = 0; i < n; i++){ normData[i] = (data[i] - means[i]) / stdevs[i]; }
			dataSet.add(normData);
		}
		
		// Training
		double[] errors = new double[epNum];
		for(int ep = 0; ep < epNum; ep++){
			Collections.shuffle(dataSet);
			double E = 0;
			for(double[] data : dataSet){
				// Forward pass
				SimpleMatrix
					x = new SimpleMatrix(n, 1, true, data),
					h = wHid.mult(addBias(x)),
					y = wOut.mult(addBias(h));
				
				// Backward pass
				SimpleMatrix
					wOutUnbias = wOut.extractMatrix(0, n, 0, p),
					deltaOut = x.minus(y),
					deltaHid = wOutUnbias.transpose().mult(deltaOut);
				wOut = wOut.plus(alpha, deltaOut.mult(addBias(h).transpose()));
				wOut = wOut.divide(wOut.normF());
				wHid = wHid.plus(alpha, deltaHid.mult(addBias(x).transpose()));

				E += getErr(x, y);
			}
			errors[ep] = E;
		}
		return errors;
	}

	public List<double[]> reconstruct(List<double[]> dataSet){
		List<double[]> res = new ArrayList<double[]>(dataSet.size());
		for(double[] data : dataSet){
			// Normalization
			double[] normData = new double[n];
			for(int i = 0; i < n; i++){ normData[i] = (data[i] - means[i])  / stdevs[i]; }
			// Forward pass
			SimpleMatrix
				x = new SimpleMatrix(n, 1, true, data),
				h = wHid.mult(addBias(x)),			// Principal components
				y = wOut.mult(addBias(h));			// Reconstruction
			res.add(y.getMatrix().getData());
		}
		return res;
	}
	
	private SimpleMatrix addBias(SimpleMatrix x){
		int xRows = x.getMatrix().getNumRows();
		SimpleMatrix xb = new SimpleMatrix(xRows+1, 1);
		xb.insertIntoThis(0, 0, x);
		xb.set(xRows, 1);
		return xb;
	}
	
	private double getErr(SimpleMatrix x, SimpleMatrix y){
		SimpleMatrix dif = x.minus(y);
		return dif.transpose().mult(dif).get(0);
	}

}