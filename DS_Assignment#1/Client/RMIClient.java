import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Scanner;
import java.awt.Point;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class RMIClient {
	   public static String generateBatch(String fileName, float writingPercentage,int numberOfNodes , int batchSize){
	   	String batch ="";
	   	Random random =new Random();
	   	int writingPercentageInt = (int)(writingPercentage *100);
	   	for (int i =0 ; i < batchSize;i++) {
			int nodeA = random.nextInt(numberOfNodes + 1)+1;
			int nodeB = random.nextInt(numberOfNodes + 1)+1;
			int rand = random.nextInt(100);
			String line = Integer.toString(nodeA) + " " + Integer.toString(nodeB);
			if (rand < writingPercentageInt) {
				if (random.nextInt(2) == 1) {
					line = "A " + line;
				} else {
					line = "D " + line;
				}
			} else {
				line = "Q " + line;
			}
			batch += line + "\n";
		}
	   	batch +="F";
	   	System.out.println(batch);
	   	//writeFile(batch,fileName);
		   return batch;

	   }
	   public static ArrayList<Point> readGraph(String fileName) {
			BufferedReader reader;
			ArrayList<Point> graph = new ArrayList<Point>();
			try {

				reader = new BufferedReader(new FileReader(
						fileName));
				String line = reader.readLine();
				while (line != null && !line.equals("S")) {
					String[] nums = line.split(" ");
					graph.add(new Point(Integer.parseInt(nums[0]),Integer.parseInt(nums[1])));
					line = reader.readLine();
				}
			
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e){
			e.printStackTrace();
		}
				return graph;
	   
	   }
	   public static void main(String[] args) {
           System.setProperty("java.security.policy","client.policy");
           Scanner in = new Scanner(System.in);
		   if (System.getSecurityManager() == null) {
	            System.setSecurityManager(new SecurityManager());
	        }

	        try {
	            String name = "graph";
							System.setProperty("java.rmi.server.hostname","localhost");
	            Registry registry = LocateRegistry.getRegistry("localhost");
	            QueriesInterface comp = (QueriesInterface) registry.lookup(name);
	            boolean flag = true;
            	System.out.println("Welcome");



while(flag) {
		            String response = "";
		            int node1=0;int node2=0;
	            	System.out.println("\n0-to use auto generated batch \n1- to add edge between two nodes enter\n2-to delete edge between two nodes enter \n3-Shortest path (Dijkstra) \n4- Shortest path (BFS)\n5- Shortest path (BellmanFord)\n6- To add initial graph from file\nPress CTRL+C to close");
	            	int choice = in.nextInt();


	  	switch (choice) {

					case 0:// adding edge to the graph  between two nodes
						System.out.println("\nenter the percentage of writing lines, 0.3 for 30% for example , \n");
						float c = in.nextFloat();
						System.out.println("\nenter the batch size , \n");
						int bs = in.nextInt();
						System.out.println("\nenter this batch upperbound , \n");
						int max = in.nextInt();
						String batch = generateBatch("batch.txt", c, max,bs);
						System.out.println("Response :");
						try {
							System.out.println(comp.executeBatch(batch));
						}catch (Exception e){
							e.printStackTrace();
						}

						break;
					case 1:// adding edge to the graph  between two nodes
						System.out.println("\nEnter the id of the first node:\n");
						node1 = in.nextInt();
						System.out.println("\nEnter the id of the second node:\n");
						node2 = in.nextInt();
						comp.addEdge(node1, node2);//implement the method on server
						response = "Done!";
						break;

					case 2://remove edge from the graph between two nodes
						System.out.println("\nEnter the id of the first node:\n");
						node1 = in.nextInt();
						System.out.println("\nEnter the id of the second node:\n");
						node2 = in.nextInt();
						comp.deleteEdge(node1,node2);//implement the method on server
						response = "Done!";
						break;

					case 3://calculate the shortest path between two nodes
						System.out.println("\nEnter the id of the first node:\n");
						node1 = in.nextInt();
						System.out.println("\nEnter the id of the second node:\n");
						node2 = in.nextInt();
						response = Integer.toString(comp.shortestPath(node1,node2,1));
						break;
						// new function from zoz
					case 4://calculate the shortest path between two nodes
						System.out.println("\nEnter the id of the first node:\n");
						node1 = in.nextInt();
						System.out.println("\nEnter the id of the second node:\n");
						node2 = in.nextInt();
						response = Integer.toString(comp.shortestPath(node1,node2,3));
						break;
					case 5://calculate the shortest path between two nodes
						System.out.println("\nEnter the id of the first node:\n");
						node1 = in.nextInt();
						System.out.println("\nEnter the id of the second node:\n");
						node2 = in.nextInt();
						response = Integer.toString(comp.shortestPath(node1,node2,2));
						break;

					case 6://add Initial graph (reading input file)
						comp.init(readGraph("inputGraph.txt"));

						break;
					case 7: // report mode
						String total = "";
						for (int i=1;i<10;i++) { // percentage
							comp.init(readGraph("inputGraph.txt"));
							long res = comp.report(generateBatch("",(float) (0.1*i), 5, 20));
							total += (String.valueOf(i)+" "+String.valueOf(res)+"\n");
						}
						writeFile(total,"percent.txt");
						total = "";
						for (int j=1;j<20;j++) {//batch size
							comp.init(readGraph("inputGraph.txt"));
							long res = comp.report(generateBatch("",(float) 0.3, 5, j));
							total += (String.valueOf(j)+" "+String.valueOf(res)+"\n");
						}
						writeFile(total,"batchsize.txt");
						total = "";
						for (int k=2;k<15;k++) {//num of nodes
							String batch1 = generateBatch("",(float) 0.3, k, 10);
							comp.init(readGraph("src/graphs/graph"+String.valueOf(k)+".txt"));
							long res = comp.report(batch1);
							total += (String.valueOf(k)+" "+String.valueOf(res)+"\n");
						}
						writeFile(total,"numofnodes.txt");
						break;
					default://close the program
						flag = false;
						break;
					}

		            //System.out.print("\033[H\033[2J");
		           // System.out.flush();
		            System.out.println( "The return value from the server is: " + response );
								System.out.println("Graph:" +comp.getGraph() );

	    }


	        } catch (Exception e) {
	            System.err.println( "Exception while getting response." );
	            e.printStackTrace();
	        }

	   }
	private static void writeFile(String data,String fileName) {
		// TODO Auto-generated method stub
		   try {
			   FileWriter myWriter = new FileWriter(fileName);
			   myWriter.write(data);
			   myWriter.close();
		   } catch (IOException e) {
			   System.out.println("An error occurred.");
			   e.printStackTrace();
		   }
	}

}
