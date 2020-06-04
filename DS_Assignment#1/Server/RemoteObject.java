import java.awt.Point;
import java.rmi.RemoteException;
import java.util.ArrayList;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.jgrapht.alg.interfaces.ShortestPathAlgorithm;
//import org.jgrapht.alg.shortestpath.BFSShortestPath;
import org.jgrapht.alg.shortestpath.BellmanFordShortestPath;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.Graph;

public class RemoteObject implements QueriesInterface {

    ReadWriteLock Lock = new ReentrantReadWriteLock();
    Graph<Integer, DefaultEdge> g = new DefaultDirectedGraph<Integer, DefaultEdge>(DefaultEdge.class);

    
	public void addEdge(int node1, int node2) throws RemoteException {
		Lock.writeLock().lock();
		if (!g.containsVertex(node1)){
			g.addVertex(node1);
		}
		if (!g.containsVertex(node2)){
			g.addVertex(node2);
		}
		
		if (node1 != node2) {
			g.addEdge(node1, node2);					
		}
		Lock.writeLock().unlock();
	}

	public void deleteEdge(int node1, int node2) throws RemoteException {
		Lock.writeLock().lock();
		g.removeEdge(node1, node2);
		Lock.writeLock().unlock();
	}

// used dijkstra's algorithm online to return the cost of shortest path, nasser the required is 'int=' not 'string'
	public int shortestPath(int node1, int node2,int option) throws RemoteException {
		int path = -1;
		switch (option) {
		case 1:
			path = Dijkstra(node1,node2);
			break;
		case 2:
			path = BellmanFord(node1,node2);
			break;
		case 3:
			path = BFS(node1,node2);			
			break;

		}
		return path;

	}
    private int BFS(int node1, int node2) {
		// TODO Auto-generated method stub
		//BFSShortestPath<Integer, DefaultEdge> BFSAlg = new BFSShortestPath<>(g);
        //ShortestPathAlgorithm.SingleSourcePaths<Integer, DefaultEdge> iPaths = BFSAlg.getPaths(node1);
    
        return 0;
	}

	private int BellmanFord(int node1, int node2) {
		// TODO Auto-generated method stub
		BellmanFordShortestPath<Integer, DefaultEdge> bellalgo = new BellmanFordShortestPath<>(g);
        ShortestPathAlgorithm.SingleSourcePaths<Integer, DefaultEdge> iPaths = bellalgo.getPaths(node1);
        
        return iPaths.getPath(node2).getLength();
	}

	private int Dijkstra(int node1, int node2) {
	// TODO Auto-generated method stub
		DijkstraShortestPath<Integer, DefaultEdge> dijkstraAlg = new DijkstraShortestPath<>(g);
        ShortestPathAlgorithm.SingleSourcePaths<Integer, DefaultEdge> iPaths = dijkstraAlg.getPaths(node1);
    
        return iPaths.getPath(node2).getLength();
}

	public String executeBatch(String s)throws RemoteException{
	    String res = "";
	    String []q = s.split("\n");
	    for (int i =0 ; i<q.length ;i++){
	        if(q[i].equals("F")){
	            break;
            }
                String[] cs = q[i].split(" ");
                if (cs[0].equals("A")) {
                    this.addEdge(Integer.parseInt(cs[1]), Integer.parseInt(cs[2]));
                } else if (cs[0].equals("D")) {
                    this.deleteEdge(Integer.parseInt(cs[1]), Integer.parseInt(cs[2]));
                } else {
                    res += this.shortestPath(Integer.parseInt(cs[1]), Integer.parseInt(cs[2]),1);
                    res += "\n";
                }

        }
	    return res;
    }



	public String getGraph(){
        String ret = "";
        for(DefaultEdge e : g.edgeSet()){
            ret += (g.getEdgeSource(e) + " --> " + g.getEdgeTarget(e));
        }
        return ret;
    }
	public void init(ArrayList<Point> data) throws RemoteException {
		// TODO Auto-generated method stub	
		for (int i=0;i<data.size();i++) {
			Point tmp = data.get(i);
			this.addEdge(tmp.x,tmp.y);
		}
				
	}
	public long report(String batch) throws RemoteException {
		// TODO Auto-generated method stub
		long startTime = System.nanoTime();
		this.executeBatch(batch);
		long endTime   = System.nanoTime();
		long totalTime = endTime - startTime;
		return totalTime;
	}




}
