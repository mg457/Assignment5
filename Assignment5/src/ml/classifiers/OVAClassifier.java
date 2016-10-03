//Nick Reminder, Maddie Gordon
//cs158 ps5
package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

/**
 * One-vs-all classifier class.
 *
 * @author Nick Reminder, Maddie Gordon
 */
public class OVAClassifier implements Classifier {

    ClassifierFactory factory;
    HashMap<Double, Classifier> classifiers;

    /**
     * Constructor for OVA classifier.
     *
     * @param factory Factory specifiying details for OVA classifier.
     */
    public OVAClassifier(ClassifierFactory factory) {
        this.factory = factory;
    }

    /**
     * Method to train OVA classifier on given data set.
     *
     * @param data Training data set.
     */
    @Override
    public void train(DataSet data) {

        Set<Double> labels = data.getLabels();
        classifiers = new HashMap<Double, Classifier>();

        //for each label, run through all examples & make labels binary
        //train a classifier on this data and add it to our map
        for (double i : labels) {
            DataSet copy = new DataSet(data.getFeatureMap());
            ArrayList<Example> examples = data.getData();
            Classifier myClassifier = factory.getClassifier();
            for (Example ex : examples) {
                if (ex.getLabel() == i) {
                    ex.setLabel(1.0);
                } else {
                    ex.setLabel(-1.0);
                }
                copy.addData(ex);
            }
            myClassifier.train(copy);
            classifiers.put(i, myClassifier);
        }
    }

    /**
     * Classifies an example based on a trained OVA classifier. Classifiers are traversed in the same order they were
     * constructed in.
     *
     * @param example Example to be classified.
     * @return Double corresponding to the predicted label.
     */
    @Override
    public double classify(Example example) {

        //get possible labels. store the labels and confidence values for the most confident positive
        //prediction and the least confident negative prediction
        Set<Double> myLabels = classifiers.keySet();
        double myMaxConfPos = 0;
        double myMaxPosLabel = -1;
        double myLeastConfNeg = 0;
        double myMinNegLabel = -1;

        for (double label : myLabels) {
            //get the relevant classifier
            Classifier myClassifier = classifiers.get(label);
            double myPrediction = myClassifier.classify(example);
            double myConfidence = myClassifier.confidence(example);

            //if positive prediction and highest confidence so far, update our stored values
            if (myPrediction > 0 && myConfidence > myMaxConfPos) {
                myMaxConfPos = myConfidence;
                myMaxPosLabel = label;
                //or, if we have the least confident negative prediction so far, update our stored values
            } else if (myConfidence < myLeastConfNeg) {
                myLeastConfNeg = myConfidence;
                myMinNegLabel = label;
            }
        }

        //return the max positive label if we actually updated it; return the min negative label if we
        //never saw a positive label
        if (myMaxPosLabel >= 0) {
            return myMaxPosLabel;
        } else return myMinNegLabel;
    }

    /**
     * Method calculating the OVA classifier's confidence in a prediction for a given example.
     *
     * @param example Example for which confidence will be evaluated.
     * @return Double corresponding to the confidence in the classifier's prediction.
     */
    @Override
    public double confidence(Example example) {
        return 0;
    }

}
