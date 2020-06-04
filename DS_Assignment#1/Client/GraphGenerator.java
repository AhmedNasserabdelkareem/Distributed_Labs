import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
public class GraphGenerator {
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	   	for (int i=1;i<=15;i++) {
		   	constructGraph(i);	   		
	   	}

	}
	
	public static void constructGraph(int Num) {
		ArrayList<ArrayList<Integer>> graph = new ArrayList<ArrayList<Integer>>();
		String data = "";
		for (int j=0;j<Num;j++) {
			graph.add(new ArrayList<Integer>());
		}
	   	Random random =new Random();
	   	for (int i=2;i<2*Num;i++) {
	   		int n1 = random.nextInt(Num)+1;
	   		int n2 = random.nextInt(Num)+1;
	   		System.out.println(graph.size()+" "+ n1+" "+ n2);
	   		if (!graph.get(n1-1).contains(n2)) {
	   			graph.get(n1-1).add(n2);
	   			data+= (String.valueOf(n1)+" "+String.valueOf(n2)+"\n");
	   		}
	   	}
	   	data+="S";
	   	writeFile(data,"src/graphs/graph"+String.valueOf(Num)+".txt");
	}
	private static void writeFile(String data,String fileName) {
		// TODO Auto-generated method stub
		   try {
			   File file = new File(fileName); 
			   file.createNewFile();
			   FileWriter myWriter = new FileWriter(file);
			   myWriter.write(data);
			   myWriter.close();
		   } catch (IOException e) {
			   System.out.println("An error occurred.");
			   e.printStackTrace();
		   }
	}

}
