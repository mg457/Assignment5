package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import java.util.*;

public class OVAClassifier implements Classifier {
	
	ClassifierFactory factory;
	
	public OVAClassifier(ClassifierFactory factory) {
		this.factory = factory;	
	}
	
	
	@Override
	public void train(DataSet data) {
		//output list of k-1/2
		//input = samples x, list of labels where y is label for x_i
		//for each k in labels, construct new label vector where z_i = 1 if y=k and z = 0 otherwise
		
		//HashMap to store classifiers <sampleIndex, label> 
		HashMap<Integer, Double> map = new HashMap<Integer, Double>();
		DataSet copy = new DataSet(data.getFeatureMap());
		
		
		

	}

	@Override
	public double classify(Example example) {
		// TODO Auto-generated method stub
		//classify: if classifier doesn't provide confidence & there is ambiguity, pick majority in conflict
		//otherwise pick most confident positive
		//if none vote positive, pick least confident negative
		return 0;
	}

	@Override
	public double confidence(Example example) {
		// prediction = b + sum w_i * f_i
		// but return 0 for now..?
		return 0;
	}

}
