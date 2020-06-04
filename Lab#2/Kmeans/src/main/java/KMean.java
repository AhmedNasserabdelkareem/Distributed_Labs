import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMean {
	private final static int DIM = 4;

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		
		if(args.length != 3) {
			System.out.println("length"+args.length+" "+args[2]);
			System.out.println("Args: input_path outputpath k");
			System.exit(-1);
		}
		
		int k = Integer.valueOf(args[2]) ;		
		Job init = Job.getInstance() ;
		
		Configuration init_conf = init.getConfiguration();
		init_conf.set("k", args[2]);
		
		init.setJarByClass(KMean.class);
		init.setJobName("clustered kmeans");
		init.setMapperClass(KMapper.class);
		init.setReducerClass(KReducer.class);
		init.setOutputKeyClass(IntWritable.class);
		init.setOutputValueClass(Text.class);
		
		
		FileInputFormat.addInputPath(init, new Path(args[0]));
		FileOutputFormat.setOutputPath(init, new Path(args[1] + "Iteration" + Integer.toString(0)));
		
		init.waitForCompletion(true);
		
		double[][] old_centroids = new double[k][4] ;

		int fi = 1 ;
		
		long t1 =  System.currentTimeMillis() ;
		
		while(true) {			
			Job job = Job.getInstance() ;
			Configuration conf = job.getConfiguration() ;
			conf.set("k", args[2]);
			
			job.setJarByClass(KMean.class);
			job.setJobName("clustered kmeans");
			job.setMapperClass(MapperWritter.class);
			job.setReducerClass(ReducerWriter.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			
			String uri =  args[1] + "Iteration" + Integer.toString(fi-1) +"/part-r-00000";
			Configuration temp_conf = new Configuration();
			FileSystem fs = FileSystem.get(URI.create(uri), temp_conf); 

			Path input_path = new Path(uri);
			FSDataInputStream input_stream = fs.open(input_path);
			BufferedReader input_buffer = new BufferedReader(new InputStreamReader(input_stream));
	
			double total_dis = 0 ;
			
			double[][] new_centroids = new double[k][DIM] ;
			for(int i = 0 ; i < k ; i++) {	
				String line = input_buffer.readLine() ;
				if(line == null) {
					for(int j = 0 ; j < DIM ; j++) {
						new_centroids[i][j] = Double.valueOf(old_centroids[i][j]) ;
					}
					continue ;
				}	
				int key = Integer.valueOf(line.split("\t")[0]) ;
				String[] new_centroid = line.split("\t")[1].split(",") ;
				for(int j = 0 ; j < DIM ; j++) {
					new_centroids[key][j] = Double.valueOf(new_centroid[j]) ;
					total_dis += Math.pow(new_centroids[key][j] - old_centroids[key][j], 2) ;
				}
				conf.set("centroid" + key, line.split("\t")[1]);
			}
			
			double threshold = Math.pow(0.001 ,2) * k * DIM  ;
			
			if(total_dis < threshold)
				break ;
			
			FileInputFormat.addInputPath(job, new Path(args[0]));
			FileOutputFormat.setOutputPath(job, new Path(args[1] + "Iteration" + Integer.toString(fi)));
			
			job.waitForCompletion(true);
			old_centroids = new_centroids;
			fi++ ;
		}

		long t2 =  System.currentTimeMillis() ;
		System.out.println("\nThe time token: " + (t2-t1)/1000.0 + "s");
	}
	
}