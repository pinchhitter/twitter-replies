package cdac.in.twitter;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URL;
import java.net.URLConnection;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.CoreEntityMention;

class SortMapByValue {

	public static Map<String, Integer> sortByComparator(Map<String, Integer> unsortMap, final boolean order){

		List<Entry<String, Integer>> list = new LinkedList<Entry<String, Integer>>(unsortMap.entrySet());

		Collections.sort(list, new Comparator<Entry<String, Integer>>() {
				public int compare(Entry<String, Integer> o1, Entry<String, Integer> o2) {
				if (order) {
				return o1.getValue().compareTo(o2.getValue());
				}
				else {
				return o2.getValue().compareTo(o1.getValue());
				}
				}
				});
		Map<String, Integer> sortedMap = new LinkedHashMap<String, Integer>();
		for (Entry<String, Integer> entry : list) {
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		return sortedMap;
	}
}

class Replies{

	String authorizationBearer = "";
	StanfordCoreNLP pipeline = null;
	Map<String, TreeMap<String, Integer>> NECount;

	Replies(){
		NECount = new TreeMap<String, TreeMap<String, Integer>>();
		try{
			BufferedReader br = new BufferedReader( new FileReader( new File( "../properties.txt") ) );
                        String ApiKey = br.readLine().split(":")[1].trim();
                        String SecretKey = br.readLine().split(":")[1].trim();
                        String Accesstoken = br.readLine().split(":")[1].trim();
                        String AccessTokenSecret = br.readLine().split(":")[1].trim();
                        authorizationBearer = br.readLine().split(":")[1].trim();

		}catch(Exception e){
			e.printStackTrace();
		}
	
	}

	public static String expand(String shortenedUrl){

		try{
			HttpURLConnection connection = (HttpURLConnection) new URL( shortenedUrl ).openConnection(Proxy.NO_PROXY);
                	connection.setInstanceFollowRedirects( false );
                	connection.getInputStream().read();
			String expandedURL = connection.getHeaderField("Location");
			if( expandedURL != null )
				return expandedURL;
		}catch(Exception e){
			e.printStackTrace();
		}
		return shortenedUrl;
    	}

	List<String> cleanText(String entitis, String entityType){
		List<String> list = new ArrayList<String>();
		if( entityType.equals("URL") ){
			 String entity = expand(  entitis );
			 list.add( entity );
		}else if ( entityType.equals("HANDLE") ){
			String[] token = entitis.split("\\s");
			list = Arrays.asList( token );
		}else{
			list.add( entitis.toLowerCase() );
		}
	return list;	
	}

	void addNE(CoreEntityMention em){
		TreeMap<String, Integer> entityCount = NECount.get( em.entityType() );
		if( entityCount == null )
			entityCount  = new TreeMap<String, Integer>();

		List<String> entities = cleanText( em.text(), em.entityType() ) ;

		for(String entity: entities ){
			Integer count = entityCount.get( entity );
			if( count == null )
				count = 0;
			count++;
			entityCount.put( entity, count );
		}

		NECount.put( em.entityType(), entityCount );
	}

	void print(){
		for(String entityType: NECount.keySet() ){
			Map<String,Integer> sortedMap =  SortMapByValue.sortByComparator( NECount.get( entityType ) , false);
			int count = 0;
			for(String entity: sortedMap.keySet() ){
				System.out.println( entityType+", "+entity+", "+ sortedMap.get( entity ) );
				count++;
				if( count ==  10)
					break;
			}
		}
	}

	void initCoreNLP(){
                Properties props = new Properties();
                props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, sentiment");
                pipeline = new StanfordCoreNLP( props );
        }

	List<String> getReplies(String conversation_id, int size){

		List<String> reply = new ArrayList<String>();
		String[] cmd = new String[] {"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query=conversation_id:"+conversation_id+"&tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id", "--header", "Authorization:Bearer "+authorizationBearer};
		int count = 0;
		try{
			while( true ){

				try{
					ProcessBuilder builder = new ProcessBuilder( cmd );
					Process process = builder.start();
					BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
					String replies = "";
					String line = null;
					while ((line = reader.readLine()) != null) {
						replies += line;
					}
					process.waitFor();
					if( replies.trim().length() == 0 )
						break;

					JSONObject obj = ( JSONObject ) ( new JSONParser().parse( replies ) );
					String nToken = (String )( (JSONObject) obj.get("meta") ).get("next_token");
					JSONArray arr = (JSONArray) obj.get("data");
					Iterator itr = arr.iterator();
					while ( itr.hasNext() ) {
						Map<String, String> map = (Map <String, String> ) itr.next();
						reply.add( map.get( "text") );
						count++;
					}
					System.err.println("replies count: "+count);
					if( count >= size)
						break;
					cmd = new String[]{"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query=conversation_id:"+conversation_id+"&tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id&next_token="+nToken, "--header", "Authorization:Bearer "+authorizationBearer};

				}catch(Exception e){
					cmd = new String[]{"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query=conversation_id:"+conversation_id+"&tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id&next_token="+(count++), "--header", "Authorization:Bearer "+authorizationBearer};
			
				}

			}
		}catch(Exception e){
			e.printStackTrace();
		}
	return reply;
	}		

	void analyser(List<String> replies){
		initCoreNLP();
		for(String reply: replies){
			CoreDocument doc = new CoreDocument( reply );
			pipeline.annotate( doc );
			for (CoreEntityMention em : doc.entityMentions()){
				addNE( em );
			}
    			//System.out.println("---");
    			//System.out.println("tokens and ner tags");
    			//String tokensAndNERTags = doc.tokens().stream().map(token -> "("+token.word()+","+token.ner()+")").collect( Collectors.joining(" "));
    			//System.out.println(tokensAndNERTags);
			//System.out.print(reply);
		}
		print();
	}

	public static void main(String[] args) throws Throwable {

		Replies reply = new Replies();
		String conversation_id = "1543099230423642112";
		int i = 0;
		int size = 100;

		while( i < args.length ){
			if( args[i].equals("-cid") || args[i].equals("-c") )
				conversation_id = args[ ++i ];
			if( args[i].equals("-s") )
				size = Integer.parseInt( args[ ++i ] );
			i++;
		}

		if( conversation_id == null || size ==  -1){
			System.err.println( "-c [conversation_id] -s [size]");
			System.exit(0);
		}

		reply.analyser(  reply.getReplies( conversation_id, size ) );
	}
}

