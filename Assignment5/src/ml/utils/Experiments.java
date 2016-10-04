package ml.utils;

import ml.classifiers.*;
import ml.data.*;

public class Experiments {
	
	public static void main(String[] args) {
		String wineFile = "/Users/maddie/Documents/FALL2016/MachineLearning/hw5/wines.train.txt";
		DataSet wineDataset = new DataSet(wineFile, DataSet.TEXTFILE);
		
		//Q1
		DecisionTreeClassifier dt = new DecisionTreeClassifier();
		dt.setDepthLimit(0);
		dt.train(wineDataset);
		//System.out.println(dt.toString());
		
		double acc = 0.0;
		System.out.println(dt.classify(wineDataset.getData().get(2)));
		for (Example ex: wineDataset.getData()) {
			if(ex.getLabel() == dt.classify(ex)){
				acc += 1.0/(double)wineDataset.getData().size();
			}	
		}
		System.out.println(acc);
		
		//Q3
//		DataSetSplit ds = wineDataset.split(0.8);
//		for(int i = 0; i < 50; i++) {
//			DecisionTreeClassifier dt = new DecisionTreeClassifier();
//			dt.setDepthLimit(i);
//			dt.train(ds.getTrain());
//			System.out.println("Depth = " + i);
//			System.out.println("Train acc: " + getAccuracy(dt, ds.getTrain(), wineDataset));
//			System.out.println("Test acc: " + getAccuracy(dt, ds.getTest(), wineDataset) + "\n");
//		}
		
		//Q4 
		
		CrossValidationSet cvs = new CrossValidationSet(wineDataset, 10, true);
		for (int i = 0; i < cvs.getNumSplits(); i++) {
			DataSetSplit dss = cvs.getValidationSet(i);
			//System.out.print(i + ",");
			
			for(int d = 1; d < 4; d++) {
				ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, d);
				OVAClassifier oc1 = new OVAClassifier(factory); //dt classifier
				AVAClassifier ac = new AVAClassifier(factory);
				//DecisionTreeClassifier dtc = (DecisionTreeClassifier) factory.getClassifier(); //get a new DTC
				//dtc.setDepthLimit(d);
				oc1.train(dss.getTrain());
				//ac.train(dss.getTrain());
				double ocAcc = getAccuracy(oc1, dss.getTest(), wineDataset);
				//double acAcc = getAccuracy(ac, dss.getTest(), wineDataset);
				System.out.print(d + ", " + ocAcc + ", " );//+ acAcc);
			}	
			System.out.println("");
		}
			
	}
	
	public static double getAccuracy(Classifier dt, DataSet ds, DataSet examples ) {
		double acc = 0.0;
		for(Example ex: ds.getData()) {
			if(ex.getLabel() == dt.classify(ex)) {
				acc += 1.0; //(double)examples.getData().size();
			}
		}
		return acc/(double)ds.getData().size();
	}
		
}
