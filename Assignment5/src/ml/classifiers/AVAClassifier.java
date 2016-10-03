//Nick Reminder, Maddie Gordon
//cs158 ps5
package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.ArrayList;
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
        ArrayList<Double> labels = (ArrayList<Double>) data.getLabels();
        //classifiers = new HashMap<Double[], Classifier>();
        classifiers = new ArrayList<Classifier>();


        //for each pair of labels, train a classifier to distinguish between the 1st and 2nd label
        for (double label1 : labels) {
            for (double label2 : labels.subList(labels.indexOf(label1) + 1, labels.size())) {
                DataSet copy = new DataSet(data.getFeatureMap());
                ArrayList<Example> examples = data.getData();
                Classifier myClassifier = factory.getClassifier();
                for (Example ex : examples) {
                    if (ex.getLabel() == label1) { //set all examples labeled with label 1 as positive
                        ex.setLabel(1.0);
                        copy.addData(ex);
                    } else if (ex.getLabel() == label2) { //set all examples labeled with label 2 as negative
                        ex.setLabel(-1.0);
                        copy.addData(ex);
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
                labelTotals.set(label1, labelTotals.get(label1) + weight);
                labelTotals.set(label2, labelTotals.get(label2) - weight);
                indexHolder++;
            }
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
