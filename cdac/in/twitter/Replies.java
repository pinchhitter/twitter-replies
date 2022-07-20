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
import edu.stanford.nlp.ling.*;

class SortMapByValue {

	public static Map<String, Original> sortByComparatorOriginal(Map<String, Original> unsortMap, final boolean order){

		List<Entry<String, Original>> list = new LinkedList<Entry<String, Original>>(unsortMap.entrySet());

		Collections.sort(list, new Comparator<Entry<String, Original>>() {
				public int compare(Entry<String, Original> o1, Entry<String, Original> o2) {
				if (order) {
				return o1.getValue().count.compareTo(o2.getValue().count);
				}
				else {
				return o2.getValue().count.compareTo(o1.getValue().count);
				}
				}
				});
		Map<String, Original> sortedMap = new LinkedHashMap<String, Original>();
		for (Entry<String, Original> entry : list) {
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		return sortedMap;
	}

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

class Original{
	String original;
	Integer count;

	Original(String original){
		this.original = original;
		this.count = 0;	
	}
}

class Replies{

	String authorizationBearer = "";
	StanfordCoreNLP pipeline = null;
	Map<String, TreeMap<String, Integer>> NECount;
	Map<String, TreeMap<String, Original>> NNPCount;
	Map<String,String> tags;
	Set<String> stopWords;

	Replies(){

		NECount = new TreeMap<String, TreeMap<String, Integer>>();
		NNPCount = new TreeMap<String, TreeMap<String, Original>>();
		tags = new TreeMap<String, String>();
		stopWords = new TreeSet<String>();

		try{
			BufferedReader br = new BufferedReader( new FileReader( new File( "../properties.txt") ) );
                        String ApiKey = br.readLine().split(":")[1].trim();
                        String SecretKey = br.readLine().split(":")[1].trim();
                        String Accesstoken = br.readLine().split(":")[1].trim();
                        String AccessTokenSecret = br.readLine().split(":")[1].trim();
                        authorizationBearer = br.readLine().split(":")[1].trim();

			br = new BufferedReader( new FileReader( new File( "./tags.txt") ) );
			String line = null;
			while( ( line = br.readLine() ) != null ){
				tags.put( line.trim(), "TRUE" );
			}

			br = new BufferedReader( new FileReader( new File( "./stopWords.txt") ) );
			while( ( line = br.readLine() ) != null ){
				stopWords.add( line.trim().toLowerCase() );
			}

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
			//e.printStackTrace();
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

	void addNNP(String word, String type ){

		TreeMap<String, Original> entityCount = NNPCount.get( type );

		if( stopWords.contains( word.trim().toLowerCase() ) )
			return;

		if( entityCount == null )
			entityCount  = new TreeMap<String, Original>();

		Original count = entityCount.get( word.toUpperCase() );

		if( count == null )
			count = new Original( word );
		count.count++;
		entityCount.put( word.toUpperCase(), count );
		NNPCount.put( type,  entityCount);
	}

	void print( Map<String, TreeMap<String, Original>> NCount, int length, boolean flag){
		for(String entityType: NCount.keySet() ){
			Map<String, Original> sortedMap =  SortMapByValue.sortByComparatorOriginal( NCount.get( entityType ) , false);
			int count = 0;
			for(String entity: sortedMap.keySet() ){
				System.out.println( entityType+", "+sortedMap.get( entity ).original+", "+ sortedMap.get( entity ).count );
				count++;
				if( count ==  length)
					break;
			}
		}
	}

	void print( Map<String, TreeMap<String, Integer>> NCount, int length ){

		for(String entityType: NCount.keySet() ){
			Map<String,Integer> sortedMap =  SortMapByValue.sortByComparator( NCount.get( entityType ) , false);
			int count = 0;
			for(String entity: sortedMap.keySet() ){
				System.out.println( entityType+", "+entity+", "+ sortedMap.get( entity ) );
				count++;
				if( count ==  length)
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
		//String[] cmd = new String[] {"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/counts/all?query=conversation_id:"+conversation_id+"&tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id&start_time=2006-03-21T00:00:01Z&grant_type=client_credentials", "--header", "Authorization:Bearer "+authorizationBearer};

		/*
		for(String c: cmd)
			System.out.print(c+" ");
		System.out.println();
		*/

		int count = 0;
		try{
			clearScreen();
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
					double done = ( count / (double) size ) * 100;
					String strDouble = String.format("%.2f", done);
					print(1,0, "Reading (tweets: "+count+"): "+strDouble+"% Completed");

					if( count >= size)
						break;
					cmd = new String[]{"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query=conversation_id:"+conversation_id+"&tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id&next_token="+nToken, "--header", "Authorization:Bearer "+authorizationBearer};

				}catch(Exception e){
					e.printStackTrace();
					break;
			
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		System.out.println();
	return reply;
	}		

	boolean validNNP(String nnp){
		String symbols = ";,@#$%6i^&*()_+:'\"";
		if( symbols.indexOf( nnp ) >= 0 )
			return false;
	return true;
	}

	void analyser(List<String> replies, int length, boolean isNER){

		initCoreNLP();
		//clearScreen();
		double count = 0.0d;
		for(String reply: replies){

			CoreDocument doc = new CoreDocument( reply );
			pipeline.annotate( doc );
			count++;
			double done = ( count / (double) replies.size() ) * 100;
			String strDouble = String.format("%.2f", done);
			print(31,0, "Processing:("+count+" Doc) "+strDouble+"% Completed");

			for (CoreEntityMention em : doc.entityMentions()){
				addNE( em );
			}

			String NNP = "";
			String tag = "";
			String original = "";

			for (CoreLabel tok : doc.tokens()) {
				/*
				if( tok.word().trim().length() > 0)
					System.err.println( tok.tag()+" | "+tok.word());
				*/

				if( tag.trim().length() > 0 ){
					original = tag;
					tag = tag+"|"+tok.tag().trim();
				}else{ 
					tag = tok.tag().trim();
					original = tag;
				}

				if( tok.word().trim().charAt(0) == '@' ){

					addNNP( tok.word().trim(), "TWITTER" );

					if( NNP.trim().length() >  0 ){
                                                addNNP( NNP.trim(), "NER" );
                                        }
					tag = "";
					NNP = "";

				}else{
					if( tags.containsKey( tag ) ){
						NNP = NNP.trim()+" "+tok.word().trim();
					}else {
						 if( NNP.trim().length() >  0 ){
							addNNP( NNP.trim(), "NER" );
						}
						tag = "";
						NNP = "";
					}
				}
				
				/*	
				if( tags.indexOf( tok.tag() ) >= 0 && tok.word().charAt(0) == '@' ){
					addNNP( tok.word().trim(), "TWITTER" );
				}

				if( tok.tag().equals("NNP") && tok.word().charAt(0) == '@' ){
					if( NNP.trim().length() > 0)
						addNNP( NNP.trim(), "NNP" );
					addNNP( tok.word().trim(), "TWITTER" );
					NNP = "";

				}else if( ( tok.tag().equals("NNPS") ||  && validNNP( tok.word().trim() ) ) || isPresent( tok.word().trim(), "NNP")  )
					NNP = NNP.trim() + " "+tok.word().trim();
				else{
					if( NNP.trim().length() > 0)
						addNNP( NNP.trim(), "NNP" );
					NNP = "";
				}
				*/
   	 		}

			if( NNP.trim().length() > 0)
				addNNP( NNP.trim(), "NER" );

			/*

			for (CoreSentence sent : doc.sentences()) {
        			System.err.println(sent.text());
    			}

			//System.out.println("--- "+reply);
			System.out.println("tokens and ner tags");
			String tokensAndNERTags = doc.tokens().stream().map(token -> "("+token.word()+","+token.ner()+")").collect( Collectors.joining(" "));
			//System.err.println(tokensAndNERTags);
			*/
		}
		if( isNER )
			print( NECount, length );

		print( NNPCount, length, true );
	}

	void clearScreen(){
		System.out.print("\033[H\033[2J");  
		System.out.flush();  		
	}

	void print(int row,int column, String test){
		char escCode = 0x1B;
		//System.out.print(String.format("%c[%d;%df %s\n",escCode,row,column, test));
		System.out.print(String.format("%c[%d;%df\b%s",escCode,row,column, test));
	}

	public static void main(String[] args) throws Throwable {

		Replies reply = new Replies();
		String conversation_id = null;
		int i = 0;
		int size = -1;
		int length = 20;
		boolean isNER = false;

		while( i < args.length ){
			if( args[i].equals("-cid") || args[i].equals("-c") )
				conversation_id = args[ ++i ];
			if( args[i].equals("-s") )
				size = Integer.parseInt( args[ ++i ] );
			if( args[i].equals("-l") )
				length = Integer.parseInt( args[ ++i ] );
			if( args[i].equals("-n") )
				isNER = true;
			i++;
		}

		if( conversation_id == null || size ==  -1){
			System.err.println( "-c <conversation_id> -s <size> -l <length> -n [NER]");
			System.exit(0);
		}
		reply.analyser(  reply.getReplies( conversation_id, size ), length, isNER );
	}
}

