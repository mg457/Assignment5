//Nick Reminder, Maddie Gordon
//cs158 ps5
package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * All-vs-all classifier class.
 *
 * @author Nick Reminder, Maddie Gordon
 */
public class AVAClassifier implements Classifier {

    ClassifierFactory factory;
    ArrayList<Classifier> classifiers;

    /**
     * Constructor for AVA classifier.
     *
     * @param factory Factory specifying details for AVA classifier.
     */
    public AVAClassifier(ClassifierFactory factory) {
        this.factory = factory;
    }

    /**
     * Method to train AVA classifier on given data set.
     *
     * @param data Training data set.
     */
    @Override
    public void train(DataSet data) {
        ArrayList<Double> labels = new ArrayList<Double>();
       // Object[] labels = data.getLabels().toArray();
       // System.out.println(labels);
        //System.out.println(data.getLabels().toString());
        //System.out.println(data.getData().get(0).getLabel());
        for(double i : data.getLabels()) {
        	//System.out.println(i);
        	labels.add(i);
        }
        //System.out.println(labels.toString());

        //classifiers = new HashMap<Double[], Classifier>();
        classifiers = new ArrayList<Classifier>();


        //for each pair of labels, train a classifier to distinguish between the 1st and 2nd label
        for (int label1 = 0; label1 < labels.size(); label1++ ) {
        	//double label1 = labels.get(i);
        	for(int label2 = label1+1; label2 < labels.size(); label2++) {
        		//double label2 = labels.get(k);
	                DataSet copy = new DataSet(data.getFeatureMap());
	                ArrayList<Example> examples = data.getData();
	                Classifier myClassifier = factory.getClassifier();
	                for (Example ex : examples) {	
	                	//System.out.println(ex.getLabel());
	                    if (ex.getLabel() == (double) label1) { //set all examples labeled with label 1 as positive
	                    	//System.out.println("here1");
	                    	Example newEx = new Example(ex);
	                        newEx.setLabel(1.0);
	                        copy.addData(newEx);
	                    } else if (ex.getLabel() == (double) label2) { //set all examples labeled with label 2 as negative
	                    	//System.out.println("here2");
	                    	Example newEx = new Example(ex);
	                        newEx.setLabel(-1.0);
	                        copy.addData(newEx);
	                    }
	                }
	                myClassifier.train(copy);
	                classifiers.add(myClassifier);
	
            }
        }

    }

    /**
     * Classifies an example based on a trained AVA classifier. Classifiers are traversed in the same order they were
     * constructed in.
     *
     * @param example Example to be classified.
     * @return Double corresponding to the predicted label.
     */
    @Override
    public double classify(Example example) {
        int indexHolder = 0;
        ArrayList<Double> labelTotals = new ArrayList<Double>(Collections.nCopies(20, 0.0));
        for (int label1 = 0; label1 < labelTotals.size(); label1++) {
            for (int label2 = label1 + 1; label2 < labelTotals.size(); label2++) {
                Classifier myClassifier = classifiers.get(indexHolder);
                double weight = myClassifier.classify(example);
                //if y is positive, raise label1 score, lower label2 score
                //if y is negative, lower label1 score, raise label2 score
                if(weight > 0) {
                    double label1Current = labelTotals.get(label1);
                    double label2Current = labelTotals.get(label2);
	                labelTotals.set(label1, label1Current + weight);
	                labelTotals.set(label2, label2Current - weight);
                }else if (weight < 0) {
                    double label1Current = labelTotals.get(label1);
                    double label2Current = labelTotals.get(label2);
                	labelTotals.set(label1, label1Current - weight);
	                labelTotals.set(label2, label2Current + weight);
                }
                indexHolder++;
            }
            //indexHolder = 0;
        }
        return labelTotals.indexOf(Collections.max(labelTotals));
    }

    /**
     * Method calculating the AVA classifier's confidence in a prediction for a given example.
     *
     * @param example Example for which confidence will be evaluated.
     * @return Double corresponding to the confidence in the classifier's prediction; [0,1].
     */
    @Override
    public double confidence(Example example) {
        return 0;
    }

}
