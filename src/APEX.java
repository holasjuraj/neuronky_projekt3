import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;

public class APEX {
	public final int n;
	public final int p;
	public final double alpha;
	private SimpleMatrix[] w;
	private SimpleMatrix[] u;
	private SimpleMatrix eigVals;
	private double[] means;
	private double[] stdevs;
	
	public APEX(int n, int p, double alpha){
		this.n = n;
		this.p = p;
		this.alpha = alpha;
		w = new SimpleMatrix[p];
		u = new SimpleMatrix[p];
	}
	
	public void reset(){
		eigVals = new SimpleMatrix(p, 1);
		Random rand = new Random();
		for(int i = 0; i < p; i++){
			w[i] = SimpleMatrix.random(n, 1, -1, 1, rand);
			u[i] = SimpleMatrix.random(p, 1, 0, 1, rand);
			for(int j = i; j < p; j++){
				u[i].set(j, 0);
			}
		}
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
		eigVals = new SimpleMatrix(p, 1);
		for(int ep = 0; ep < epNum; ep++){
			Collections.shuffle(dataSet);
			double E = 0;
			for(double[] data : dataSet){
				SimpleMatrix x = new SimpleMatrix(n, 1, true, data);
				SimpleMatrix y = output(x);
				SimpleMatrix y_im1 = new SimpleMatrix(p, 1);
				for(int i = 0; i < p; i++){
					w[i] = w[i].plus(alpha*y.get(i), x);
					w[i] = w[i].divide(w[i].normF());
					u[i] = u[i].plus(-alpha*y.get(i), y_im1.plus(y.get(i), u[i]));
					y_im1.set(i, y.get(i));
				}
				E += getErr(x, y);
				if(ep == epNum-1){
					eigVals = eigVals.plus(y.elementPower(2));
				}
			}
			errors[ep] = E;
		}
		eigVals = eigVals.divide(dataSet.size());
		return errors;
	}
	
	public List<double[]> getPComponents(){
		List<double[]> res = new ArrayList<double[]>(p);
		for(int i = 0; i < p; i++){
			res.add(deNormalize(w[i].scale(2)));	// Scaling is only for better contrast in visualization
		}
		return res;
	}
	
	public double[] getEigValues(){
		return eigVals.getMatrix().getData();
	}

	public List<double[]> reconstruct(List<double[]> dataSet, int maxP){
		SimpleMatrix W = new SimpleMatrix(n, maxP);
		for(int i = 0; i < maxP; i++){
			W.insertIntoThis(0, i, w[i]);
		}
		List<double[]> res = new ArrayList<double[]>(dataSet.size());
		for(double[] data : dataSet){
			// Normalization
			double[] normData = new double[n];
			for(int i = 0; i < n; i++){ normData[i] = (data[i] - means[i])  / stdevs[i]; }
			// Projection to PCA space
			SimpleMatrix x = new SimpleMatrix(n, 1, true, normData);
			SimpleMatrix y = output(x).extractMatrix(0, maxP, 0, 1);
			// Transformation back to original space
			SimpleMatrix xRec = W.mult(y);
			res.add(deNormalize(xRec));
		}
		return res;
	}
	
	private SimpleMatrix output(SimpleMatrix x){
		SimpleMatrix y = new SimpleMatrix(p, 1);
		for(int i = 0; i < p; i++){
			y.set(i,  w[i].transpose().mult(x).plus(u[i].transpose().mult(y)).get(0)  );
		}
		return new SimpleMatrix(y);
	}
	
	private double getErr(SimpleMatrix x, SimpleMatrix y){
		SimpleMatrix W = new SimpleMatrix(n, p);
		for(int i = 0; i < p; i++){
			W.insertIntoThis(0, i, w[i]);
		}
		SimpleMatrix dif = x.minus(W.mult(y));
		return dif.transpose().mult(dif).get(0);
	}
	
	private double[] deNormalize(SimpleMatrix a){
		double[] res = new double[n];
		for(int i = 0; i < n; i++){
			res[i] = a.get(i)*stdevs[i] + means[i];
		}
		return res;
	}
	
	public void analyticPCA(List<double[]> dataSetOrig){
		reset();
		// normalization
		List<double[]> dataSet = new ArrayList<double[]>(dataSetOrig.size());
		means = new double[n];
		for(double[] data : dataSetOrig){
			for(int i = 0; i < n; i++){ means[i] += data[i]; }
		}
		for(int i = 0; i < n;i++){ means[i] /= dataSetOrig.size(); }
		
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
		
		// PCA
		SimpleMatrix R = new SimpleMatrix(n, n);
		for(double[] data : dataSet){
			SimpleMatrix x = new SimpleMatrix(n, 1, true, data);
			R = R.plus(x.mult(x.transpose()));
		}
		R.divide(dataSet.size());
		SimpleEVD<SimpleMatrix> evd = R.eig();
		eigVals = new SimpleMatrix(p, 1);
		for(int i = 0; i < p; i++){
			w[i] = evd.getEigenVector(i);
			w[i] = w[i].divide(w[i].normF());
			eigVals.set(i, evd.getEigenvalue(i).getReal());
			u[i] = new SimpleMatrix(p, 1);
		}
	}
	
}