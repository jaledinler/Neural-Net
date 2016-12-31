import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.instance.Randomize;

public class NNImpl {
	double l;
	int e, h;
	Instances trainSet, testSet;
	Map<String, Double> weights;
	Map<String, Double> hiddenWeights;
	Map<Integer,Double> strain;
	Map<Integer,Double> stest;
	Map<Integer,Double> hiddenOuts;
	int labelIndex;
	NNImpl(Instances trainSet, Instances testSet, double l, int h, int e) {
		this.trainSet = trainSet;
		this.testSet = testSet;
		this.e = e;
		this.h= h;
		this.l = l;
		this.labelIndex = trainSet.numAttributes()-1;
		this.weights = new TreeMap<String, Double>();
		this.hiddenWeights = new TreeMap<String, Double>();
		this.hiddenOuts = new TreeMap<Integer,Double>();
		this.strain = new TreeMap<Integer,Double>();
		this.stest = new TreeMap<Integer,Double>();

	}
	void trainMultipleLayer (Instances train, double learningRate, int epoch, int hidden){
		int size = train.numInstances();
		initializeHiddenWeights(train, hidden);
		for(int i = 0; i < epoch; i++) {
			int trainCount = 0;
			double error = 0;
			for (int j = 0; j < size; j++) {
				Randomize r = new Randomize();
				int seed = r.getRandomSeed();
				Random rand = new Random(seed);   // create seeded number generator
				Instances randData = new Instances(train);   // create copy of original data
				randData.randomize(rand); 
				Instance instance = randData.get(j);
				for(int n = 0; n < hidden; n++) {					
					String temp = String.valueOf(n);					
					double sum = hiddenWeights.get("bias");
					for (int k = 0; k < train.numAttributes(); k++){
						Attribute att = train.attribute(k);
						if (att.isNumeric()) {
							double weight = hiddenWeights.get(temp+att.name());
							standardization(randData, testSet, att);
							for(int m = 0; m < att.numValues(); m++) {								
								sum += (strain.get(j)*weight);
							}
						}
						if (att.isNominal()) {
							for (int l = 0; l < att.numValues(); l++) {
								if((int) instance.value(att) == l) {
									sum += hiddenWeights.get(temp+att.name()+String.valueOf(l));
									break;
								}
							}
							//sum += weight;
						}
					}
					double sigmoid = (double)1/(1+Math.exp(-sum));
					hiddenOuts.put(n,sigmoid);
				}
				String biasName ="bias"; 
				double biasWeight = hiddenWeights.get(biasName);
				double total = hiddenWeights.get(biasName); 
				for(int n = 0; n < hidden; n++) {					
					String temp = String.valueOf(n);
					double weight = hiddenWeights.get(temp);
					total += (weight*hiddenOuts.get(n));
				}
				total = (double)1/(1+Math.exp(-total));
				if ((int)instance.value(labelIndex) == 0)
					error += -Math.log(1-total);
				if ((int)instance.value(labelIndex) == 1)
					error += -Math.log(total);
				if (total <= 0.5 && (int) instance.value(labelIndex) == 0)
					trainCount++;
				if (total > 0.5 && (int) instance.value(labelIndex) == 1)
					trainCount++;
				if (i < epoch -1) {
					double delta = total-instance.value(labelIndex);
					biasWeight = biasWeight-learningRate*delta;
					hiddenWeights.put(biasName,biasWeight);
					for(int n = 0; n < hidden; n++) {
						String temp = String.valueOf(n); 
						double weight = hiddenWeights.get(temp);					
						double hOut = hiddenOuts.get(n);
						double newWeight = weight - learningRate*(delta)*hOut;
						hiddenWeights.put(temp, newWeight); 
						biasWeight = hiddenWeights.get(temp+"bias"); 
						biasWeight = biasWeight-learningRate*delta*weight*hOut*(1-hOut);
						hiddenWeights.put(temp+"bias",biasWeight); 
						for (int k = 0; k < train.numAttributes(); k++){
							Attribute att = train.attribute(k);
							String name = att.name();
							if(att.isNumeric()) {
								double w = hiddenWeights.get(temp+name);
								standardization(randData,testSet,att);
								double value = strain.get(j);
								newWeight = w - learningRate*delta*weight*value*hOut*(1-hOut);
								hiddenWeights.put(temp+name, newWeight);
							}
							if (att.isNominal()) {
								for (int l = 0; l < att.numValues(); l++) {
									if((int) instance.value(att) == l) {
										double w = hiddenWeights.get(temp+name+String.valueOf(l));
										newWeight = w - learningRate*delta*weight*hOut*(1-hOut);
										hiddenWeights.put(temp+name+String.valueOf(l), newWeight);
										break;
									}
								}								
							}
							
						}
					}
				}
			}
			System.out.println(String.valueOf(i+1)+"\t"+String.format("%.14f",error)+"\t"+String.valueOf(trainCount)+"\t"+String.valueOf(size-trainCount));
		}
	}
	void trainSingleLayer(Instances train, double learningRate, int epoch, int hidden){
		int size = train.numInstances();
		initializeWeights(train);
		for(int i = 0; i < epoch; i++) {
			double error = 0;
			int trainCount = 0;
			for (int j = 0; j < size; j++) {
				Randomize r = new Randomize();
				int seed = r.getRandomSeed();
				Random rand = new Random(seed);   // create seeded number generator
				Instances randData = new Instances(train);   // create copy of original data
				randData.randomize(rand); 
				Instance instance = randData.get(j);	
				double sum = weights.get("bias");
				for (int k = 0; k < train.numAttributes()-1; k++){
					Attribute att = train.attribute(k);
					if (att.isNumeric()) {
						double weight = weights.get(att.name());
						standardization(randData, testSet, att);
						double value = strain.get(j);
						sum += (value*weight);
					}
					if (att.isNominal()) {
						for(int l = 0; l < att.numValues(); l++){
							if((int) instance.value(att) == l) {
								sum += weights.get(att.name()+String.valueOf(l));
								break;
							}
						}
					}
				}
				double sigmoid = (double)1/(1+Math.exp(-sum));
				if ((int)instance.value(labelIndex) == 0)
					error += -Math.log(1-sigmoid);
				if ((int)instance.value(labelIndex) == 1)
					error += -Math.log(sigmoid);
				if (sigmoid < 0.5 && instance.value(labelIndex) == 0)
					trainCount++;
				if (sigmoid >= 0.5 && instance.value(labelIndex) == 1)
					trainCount++;

				double weight = weights.get("bias");
				double newWeight = weight - learningRate*(sigmoid-instance.value(labelIndex));
				weights.put("bias", newWeight);
				for (int k = 0; k < train.numAttributes()-1; k++){
					Attribute att = train.attribute(k);
					String name = att.name();
					if(att.isNumeric()) {
						weight = weights.get(name);
						standardization(randData, testSet, att);
						double value = strain.get(j);
						newWeight = weight - learningRate*(sigmoid-instance.value(labelIndex))*value;
						weights.put(name, newWeight);
					}
					if (att.isNominal()) {
						for(int l = 0; l < att.numValues(); l++){
							if((int) instance.value(att) == l) {
								weight = weights.get(name+String.valueOf(l));
								newWeight = weight - learningRate*(sigmoid-instance.value(labelIndex));
								weights.put(name+String.valueOf(l), newWeight);
								break;
							}
						}						
					}					
				}
			}
			System.out.println(String.valueOf(i+1)+"\t"+String.format("%.14f",error)+"\t"+"Correct: "+String.valueOf(trainCount)+"\t"+"InCorrect: "+String.valueOf(size-trainCount));
		}
	}
	void classify(Instances testSet, int hidden) {
		int testCount = 0;
		int size = testSet.numInstances();
		if(hidden == 0) {
			for (int i = 0; i < size; i++) {
				Instance instance = testSet.get(i);
				double sum = weights.get("bias");
				for (int k = 0; k < testSet.numAttributes(); k++){
					Attribute att = testSet.attribute(k);
					String name = att.name();
					if(att.isNumeric()) {
						double weight = weights.get(name);
						standardization(trainSet,testSet,att);
						sum+=(weight*stest.get(i));
					}
					if (att.isNominal()) {
						for(int l = 0; l < att.numValues(); l++){
							if((int) instance.value(att) == l) {
								double weight = weights.get(att.name()+String.valueOf(l));
								sum += weight;
								break;
							}
						}
					}
				}
				double tmp = 1+Math.exp(-sum);
				double sigmoid = (double)1/tmp;
				int predicted = 0;
				if (sigmoid > 0.5)
					predicted = 1;
				int value = (int) instance.value(labelIndex);
				if (sigmoid <= 0.5 && value == 0)
					testCount++;
				if (sigmoid > 0.5 && value == 1)
					testCount++;
				System.out.println("Activation: "+String.format("%.6f",sigmoid)+"\t"+"Predicted: "+
						String.valueOf(predicted)+"\t"+"Actual: "+String.valueOf(value));
			}			
		}
		Map<Integer, Double> hOuts = new TreeMap<Integer, Double>();
		int tpr = 0;
		int fpr = 0;
		int fnr = 0;
		int tnr = 0;
		if(hidden > 0) {
			for (int i = 0; i < testSet.numInstances(); i++) {	
				Instance instance = testSet.get(i);
				for (int j = 0; j < hidden; j++) {
					String temp = String.valueOf(j);
					double sum = hiddenWeights.get(temp+"bias"); 
					for (int k = 0; k < testSet.numAttributes(); k++){
						Attribute att = testSet.attribute(k);
						String name = att.name();
						if(att.isNumeric()) {
							standardization(trainSet, testSet, att);
							double weight = hiddenWeights.get(temp+name);
							sum += (weight*stest.get(i));
						}
						if (att.isNominal()) {
							for (int l = 0; l < att.numValues(); l++) {
								if((int)instance.value(att) == l) {
									double weight = hiddenWeights.get(temp+name+String.valueOf(l));
									sum += weight;
									break;
								}
							}
						}
					}
					double sigmoid = (double)1/(1+Math.exp(-sum));
					hOuts.put(j, sigmoid);
				}
				double sum = hiddenWeights.get("bias"); 
				for (int j = 0; j < hidden; j++) {
					String temp = String.valueOf(j);
					sum += (hOuts.get(j)*hiddenWeights.get(temp));
				}
				double sigmoid = (double)1/(1+Math.exp(-sum));
				int predicted = 0;
				if (sigmoid > 0.1)
					predicted = 1;
				int value = (int) instance.value(labelIndex);
				if (sigmoid <= 0.5 && value == 0)
					testCount++;
				if (sigmoid > 0.5 && value == 1)
					testCount++;
                if (value == 0 && predicted == 1)
                	fpr++;
                if (value == 1 && predicted == 1)
                	tpr++;
                if (value == 1 && predicted == 0)
                	fnr++;
                if (value == 0 && predicted == 0)
                	tnr++;
				
				System.out.println("Activation: "+String.format("%.6f",sigmoid)+"\t"+"Predicted: "+
						String.valueOf(predicted)+"\t"+"Actual: "+String.valueOf(value));
			}
		}
		System.out.println("Correct: "+String.valueOf(testCount)+"\t"+"Incorrect: "+String.valueOf(size-testCount));
		System.out.println("TPR: "+String.valueOf((double)tpr/(tpr+fnr))+"\t"+"FPR: "+String.valueOf((double)fpr/(tnr+fpr))+"\n");
		System.out.println("TPR: "+String.valueOf(tpr)+"\t"+"FPR: "+String.valueOf(fpr)+"\n");
		System.out.println("TNR: "+String.valueOf(tnr)+"\t"+"FNR: "+String.valueOf(fnr)+"\n");
	}
	void initializeWeights(Instances instances) {
		int size = instances.numAttributes();
		for (int i = 0; i < size; i++) {
			Attribute att = instances.attribute(i);
			if (att.isNumeric()) {
				double weight = generateWeight();
				weights.put(att.name(), weight);
			}
			if (att.isNominal()) {
				for (int j = 0; j < att.numValues(); j++) {
					double weight = generateWeight();
					weights.put(att.name()+String.valueOf(j), weight);
				}
			}
		}
		weights.put("bias", generateWeight());
	}
	void initializeHiddenWeights(Instances instances, int hidden) {
		int size = instances.numAttributes();
		for(int i = 0; i < hidden; i++){
			for (int j = 0; j < size; j++) {
				Attribute att = instances.attribute(j);
				if(att.isNumeric()) {
					double weight = generateWeight();
					hiddenWeights.put(String.valueOf(i)+att.name(), weight);
				}
				if (att.isNominal()) {
					for (int k = 0; k < att.numValues(); k++) {
						double weight = generateWeight();
						hiddenWeights.put(String.valueOf(i)+att.name()+String.valueOf(k), weight);
					}
				}
			}
			hiddenWeights.put(String.valueOf(i)+"bias", generateWeight());
		}

		for(int i = 0; i < hidden; i++){
			double weight = generateWeight();
			hiddenWeights.put(String.valueOf(i), weight);

		}
		hiddenWeights.put("bias", generateWeight());
	}
	double generateWeight() {
		double start = -0.01;
		double end = 0.01;
		double random = new Random().nextDouble();
		double weight = start + (random * (end - start));
		return weight;
	}
	void standardization (Instances train, Instances test, Attribute att) {
		double mu = 0;
		double sigma = 0;
		int size = train.numInstances();
		for(int i = 0; i < size; i++) {
			mu += train.get(i).value(att);
		}
		mu = (double)mu/size;
		for(int i = 0; i < size;i++) {
			double value = train.get(i).value(att);
			sigma += Math.pow((value-mu),2);
		}
		sigma = Math.sqrt((double) sigma/size);
		for(int i = 0; i < size;i++) {
			double value = train.get(i).value(att);
			value = (double)(value-mu)/sigma;
			strain.put(i,value);
		}
		for(int i = 0; i < test.numInstances(); i++) {
			double value = test.get(i).value(att);
			value = (double)(value-mu)/sigma;
			stest.put(i,value);
		}
	}
}