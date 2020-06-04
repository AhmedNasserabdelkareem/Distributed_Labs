import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class ReducerWriter extends Reducer<IntWritable, Text, IntWritable, Text>{

	@Override
	protected void reduce(IntWritable centroid_ind, Iterable<Text> points,Context context) throws IOException, InterruptedException {
				
		double[] average_dim = new double[4]; 
		
		int dim_n = 0 ;
		
		for(Text point_str : points) {
			
			String[] point = point_str.toString().split(",") ;
			for(int i = 0 ; i < point.length-1 ; i++) {
				double point_dim = Double.valueOf(point[i]) ;
				average_dim[i] = (dim_n * average_dim[i] + point_dim) / (dim_n + 1)  ;
			}
			dim_n++ ;
		}
		
		StringBuilder str_b = new StringBuilder() ;
		for(int i = 0 ; i < average_dim.length ; i++) {
			str_b.append(String.valueOf(average_dim[i])) ;
			if( i != average_dim.length - 1) {
				str_b.append(",") ;
			}
		}
		
		String value = str_b.toString() ;
		context.write(centroid_ind, new Text(value));
	}

	
	
}