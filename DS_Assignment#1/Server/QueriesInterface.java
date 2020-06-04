import java.awt.Point;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.*;
/**
 *
 */

/**
 * @author Ahmednasser
 *
 */
public interface QueriesInterface extends Remote{
	   void addEdge(int firstID,int secondID) throws RemoteException;
	   void deleteEdge(int firstID,int secondID) throws RemoteException;
	   int shortestPath(int firstID,int secondID,int option) throws RemoteException;
	   String getGraph() throws RemoteException;
	   String executeBatch(String batch) throws RemoteException;//the batch is  like A 2 3\nD 4 1\nF
	   void init (ArrayList<Point> data) throws RemoteException;
	   long report(String batch) throws RemoteException;
}
