import joinery.DataFrame;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class train_c45 {
    public static void deal_data(String file_path,ArrayList<Integer> ks) throws IOException {
        ArrayList<Integer> all_index = new ArrayList<>();
        for (int i = 0; i <= 600; i++) {
            all_index.add(i);
        }
        for(int k:ks){
            for (int remain_index = (k - 3) * 100; remain_index <= (k - 2) * 100 - 1; remain_index++) {
                all_index.remove(new Integer(remain_index));
            }
        }
        all_index.remove(new Integer(600));
        DataFrame<Object> df = DataFrame.readCsv(file_path);
        Integer[] select_index = (Integer[]) all_index.toArray(new Integer[all_index.size()]);
        DataFrame<Object> df_new = df.reindex(select_index, true);
        df_new.writeCsv("src/main/java/temp.csv");
    }

    public static void main(String[] args) throws Exception {
        String species = args[0];
        ArrayList<Integer> ks = new ArrayList<>();
        for (int i = 1; i < args.length-1; i++) {
            ks.add(Integer.parseInt(args[i]));
        }
        String C = args[args.length-1];

        FileWriter writer = new FileWriter("src/main/res/res.txt");
        writer.write("ACC\tSn\tSp\tAUC\tAP\tMCC\tTP\tTN\tFP\tFN\n");
        for(int split_index = 1;split_index<=10;split_index++) {
            String classname = "weka.classifiers.trees.J48";
            String[] options = new String[2];
            options[0] = "-C";
            options[1] = C;
            Classifier classifier = (Classifier) Utils.forName(Classifier.class,classname,options);
            String path = "src/main/data/"+species+"/org_data.csv";
            deal_data(path,ks);
            File input_file = new File("src/main/java/temp.csv");
            CSVLoader loader = new CSVLoader();
            loader.setSource(input_file);
            Instances instances = loader.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);
            Evaluation evaluation = new Evaluation(instances);
            for (int fold_index = 1; fold_index <= 5; fold_index++) {
                loader = new CSVLoader();
                String path_train = "src/main/data/"+species+"/split_"+split_index+"/fold_"+fold_index+"/train.csv";
                deal_data(path_train,ks);
                input_file = new File("src/main/java/temp.csv");
                loader.setSource(input_file);
                Instances train_instances = loader.getDataSet();
                train_instances.setClassIndex(train_instances.numAttributes() - 1);

                String path_test = "src/main/data/"+species+"/split_"+split_index+"/fold_"+fold_index+"/test.csv";
                deal_data(path_test,ks);
                input_file = new File("src/main/java/temp.csv");
                loader.setSource(input_file);
                Instances test_instances = loader.getDataSet();
                test_instances.setClassIndex(test_instances.numAttributes() - 1);

                Classifier clscopy = AbstractClassifier.makeCopy(classifier);
                clscopy.buildClassifier(train_instances);
                evaluation.evaluateModel(clscopy,test_instances);
            }
            double acc = evaluation.pctCorrect();
            double sen = evaluation.truePositiveRate(0);
            double spe = evaluation.truePositiveRate(1);
            double auc = evaluation.areaUnderROC(0);
            double ap = evaluation.areaUnderPRC(0);
            double mcc = evaluation.matthewsCorrelationCoefficient(0);
            writer.write(+acc+"\t"+sen+"\t"+spe+"\t"+auc+"\t"+ap+"\t"+mcc+"\n");
        }
        writer.close();
    }
}




