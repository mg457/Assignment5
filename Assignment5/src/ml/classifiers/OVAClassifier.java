package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import java.util.*;

public class OVAClassifier implements Classifier {
	
	ClassifierFactory factory;
	HashMap<Double, Classifier> classifiers;
	
	public OVAClassifier(ClassifierFactory factory) {
		this.factory = factory;	
	}

	@Override
	public void train(DataSet data) {
		
		Set<Double> labels = data.getLabels();
		classifiers = new HashMap<Double, Classifier>();

		//for each label, run through all examples & make labels binary
		//train a classifier on this data and add it to our map
		for(double i : labels) {
			DataSet copy = new DataSet(data.getFeatureMap());
			ArrayList<Example> examples = copy.getData();
			Classifier myClassifier = factory.getClassifier();
			for (Example ex : examples) {
				if(ex.getLabel() == i) {
					ex.setLabel(1.0);
				}
				else{
					ex.setLabel(0.0);
				}
			}
			myClassifier.train(copy);
			classifiers.put(i, myClassifier);
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

		return 0;
	}

}
