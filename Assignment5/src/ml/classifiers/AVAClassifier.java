package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;

public class AVAClassifier implements Classifier {
	
	ClassifierFactory factory;
	ArrayList<Classifier> classifiers;
	
	public AVAClassifier(ClassifierFactory factory) {
		this.factory = factory;
	}

	@Override
	public void train(DataSet data) {
		ArrayList<Double> labels = (ArrayList<Double>) data.getLabels();
		//classifiers = new HashMap<Double[], Classifier>();
		classifiers = new ArrayList<Classifier>();
		
		
		//for each pair of labels, train a classifier to distinguish between the 1st and 2nd label
		for(double label1 : labels) {
			for(double label2 : labels.subList(1, labels.size())) {
				DataSet copy = new DataSet(data.getFeatureMap());
				Classifier myClassifier = factory.getClassifier();
				ArrayList<Example> examples = copy.getData();
				for(Example ex : examples) {
					if(ex.getLabel() == label1) { //set all examples labeled with label 1 as positive
						ex.setLabel(1.0);
					}else if(ex.getLabel() == label2) { //set all examples labeled with label 2 as negative
						ex.setLabel(0.0);
					}
					copy.addData(ex);
				}
				
				myClassifier.train(copy);
				
				//store pair of labels with associated classifier
//				Double[] comparedLabels = new Double[2];
//				comparedLabels[0] = label1;
//				comparedLabels[1] = label2;
//				classifiers.put(comparedLabels, myClassifier); 
				
				//store classifier associated with the current pair of labels
				classifiers.add(myClassifier);
			}
		}
		
	}

	@Override
	public double classify(Example example) {
		
		ArrayList<Double> labelTotals = new ArrayList<Double>(Collections.nCopies(20, 0.0)); //hard-coded for 20 labels
		
		
		//for each classifier distinguishing label_i & label_i+1, update running total for 2 labels
		for(int i = 0; i < classifiers.size(); i++) {
			Classifier c = classifiers.get(i);
			double weight1 = labelTotals.get(i);
			double weight2 = labelTotals.get(i+1);
			double y = c.classify(example);
			if(y == 1) {
				weight1 += y;
			}else{
				weight2 -= y;
			}
			labelTotals.set(i, weight1);
			labelTotals.set(i+1, weight2);
		}
		return 0.0; //label total furthest from 0?
	}

	@Override
	public double confidence(Example example) {
		return 0;
	}

}
