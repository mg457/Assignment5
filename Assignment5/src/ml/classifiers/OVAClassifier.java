package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import java.util.*;

public class OVAClassifier implements Classifier {
	
	ClassifierFactory factory;
	HashMap<Double, DecisionTreeClassifier> classifiers;
	
	public OVAClassifier(ClassifierFactory factory) {
		this.factory = factory;	
	}
	
	
	@Override
	public void train(DataSet data) {
		
		// copy data because will be changing the labels
		DataSet copy = new DataSet(data.getFeatureMap());
		
		Set<Double> labels = copy.getLabels();
		ArrayList<Example> examples = copy.getData();
		classifiers = new HashMap<Double, DecisionTreeClassifier>();
		
		
		//for each label, run through all examples & make labels binary
		for(double i : labels) {
			for (Example ex : examples) {
				DecisionTreeClassifier dtc = new DecisionTreeClassifier();
				if(ex.getLabel() == i) {
					ex.setLabel(1.0);
					dtc.train(copy);
					classifiers.put(i, dtc);
				}
				else{
					ex.setLabel(0.0);
					dtc.train(copy);
					classifiers.put(i, dtc);
				}
			}	
		}	
	}

	@Override
	public double classify(Example example) {
		//classify: if classifier doesn't provide confidence & there is ambiguity, pick majority in conflict
		//otherwise pick most confident positive
		//if none vote positive, pick least confident negative
		
		for(DecisionTreeClassifier dt : classifiers.values()) {
			
		}
		example.getLabel();
		
		return 0;
	}

	@Override
	public double confidence(Example example) {
		// prediction = b + sum w_i * f_i
		// but return 0 for now..?
		return 0;
	}

}
