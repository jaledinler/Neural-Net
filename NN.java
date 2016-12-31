import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.core.Instances;
import weka.filters.unsupervised.instance.Randomize;

public class NN {
    static int numOfAttributes;
	public static void main(String[] args) throws IOException {
		if(args.length != 5) {
			System.out.println(args.length);
			System.err.println("Usage: Java DecisionTree"
					+ "<train-set-file> <test-set-file> threshold");
			System.exit(1);;
		}
		BufferedReader train = new BufferedReader(
				new FileReader(args[0]));
		BufferedReader test = new BufferedReader(
				new FileReader(args[1]));
		Instances trainSet = new Instances(train);
		Randomize r = new Randomize();
		int seed = r.getRandomSeed();
		Random rand = new Random(seed);   // create seeded number generator
		Instances randData = new Instances(trainSet);   // create copy of original data
		randData.randomize(rand); 
		Instances testSet = new Instances(test);
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		double l = Double.parseDouble(args[2]);
		int h = Integer.parseInt(args[3]);
		int e = Integer.parseInt(args[4]);
		NNImpl nn = new NNImpl(randData,testSet, l, h, e);
	    if(h == 0)
	    	nn.trainSingleLayer(randData, l, e, h);
	    if(h > 0)
	    	nn.trainMultipleLayer(randData, l, e, h);
        nn.classify(testSet, h);
		train.close();
		test.close();

	}

}
