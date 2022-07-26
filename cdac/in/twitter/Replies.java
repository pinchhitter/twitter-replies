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

class Original{

	String original;
	Integer count;
	Long likeCount;
	Long retweetCount;		

	PriorityQueue<TEntry> priorityQueue;
	Map<String,Integer> counts;

	Original(String original){

		this.original = Replies.cleanOriginal( original );
		this.count = 0;	
		this.likeCount = (Long) 0L;
		this.retweetCount = (Long) 0L;
		this.priorityQueue = new PriorityQueue<TEntry>();
		this.counts = new TreeMap<String, Integer>();	

		this.counts.put( this.original, 1);

		this.priorityQueue.add( new TEntry( this.original, 1 ) ) ;
	}

	Original(String original, Long likeCount, Long retweetCount){

		this.original = Replies.cleanOriginal( original );
		this.count = 0;	
		this.likeCount = likeCount;
		this.retweetCount = retweetCount;
		this.priorityQueue = new PriorityQueue<TEntry>();
		this.counts = new TreeMap<String, Integer>();	

		this.counts.put( this.original, 1);
		this.priorityQueue.add( new TEntry( this.original, 1 ) ) ;
	}
	
	void add(String original){

		original = Replies.cleanOriginal( original );
		Integer count = counts.get( original );
		if( count == null )
			count = 0;
		count++;	

		TEntry entry = new TEntry( original, count );

		priorityQueue.remove( entry );
		priorityQueue.add( entry );

		counts.put( original, count);
	}

	String getOriginal(){
		return priorityQueue.peek().key;
	}

	void print(){

		Iterator<TEntry> itr = priorityQueue.iterator();
		System.err.println("--------- B -----------");
		while( itr.hasNext() ){
			TEntry e = itr.next();
			System.err.println( e.key+" <> "+e.value);
		}
		System.err.println("--------- E -----------");
	}
}

class TEntry implements Comparable<TEntry> {

	String key;
	Integer value;	
   	public TEntry(String key, Integer value) {
        	this.key = key;
        	this.value = value;
    	}

	@Override
    	public boolean equals(Object other) {
		TEntry o = (TEntry) other;
        	if (o == this) {
            		return true;
        	}else
			return o.key.equals( this.key );
	}
    	@Override
    	public int compareTo(TEntry other) {
		if( other.key.equals( this.key ) )
			return 0;
        	return other.value.compareTo( this.value );
    	}
}

class NEREntry implements Comparable<NEREntry> {

	String key;
	Original value;	

   	public NEREntry(String key, Original value) {
        	this.key = key;
        	this.value = value;
    	}

	@Override
    	public boolean equals(Object other) {
		NEREntry o = (NEREntry) other;
        	if (o == this) {
            		return true;
        	}else
			return o.key.equals( this.key );
	}
    	@Override
    	public int compareTo(NEREntry other) {
		if( other.key.equals( this.key ) )
			return 0;
        	return other.value.count.compareTo( this.value.count );
    	}
}

class SortMapByValue {

	public static Map<String, Original> sortByComparatorOriginal(Map<String, Original> unsortMap, final boolean order){
		Map<String, Original> sortedMap = new LinkedHashMap<String, Original>();
		try{

			List<Entry<String, Original>> list = new LinkedList<Entry<String, Original>>(unsortMap.entrySet());

			Collections.sort( list, new Comparator<Entry<String, Original>>() {
					public int compare(Entry<String, Original> o1, Entry<String, Original> o2) {
					if (order) {
						return o1.getValue().count.compareTo(o2.getValue().count);
					}
					else {
						return o2.getValue().count.compareTo(o1.getValue().count);
					}
			}
			});
			for (Entry<String, Original> entry : list) {
				sortedMap.put(entry.getKey(), entry.getValue());
			}
		}catch(Exception e){
			// doing nothing 
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

class Replies{

	String authorizationBearer = "";
	StanfordCoreNLP pipeline = null;
	Map<String, TreeMap<String, Integer>> NECount;
	Map<String, TreeMap<String, Original>> NNPCount;
	PriorityQueue<NEREntry> toppers;

	Map<String,String> ners;
	Map<String,String> phrs;
	static Set<String> stopWords;

	Replies(){

		NECount = new TreeMap<String, TreeMap<String, Integer>>();
		NNPCount = new TreeMap<String, TreeMap<String, Original>>();
		ners = new TreeMap<String, String>();
		phrs = new TreeMap<String, String>();
		stopWords = new TreeSet<String>();
		toppers = new PriorityQueue<NEREntry>();

		try{
			BufferedReader br = new BufferedReader( new FileReader( new File( "../properties.txt") ) );
			String ApiKey = br.readLine().split(":")[1].trim();
			String SecretKey = br.readLine().split(":")[1].trim();
			String Accesstoken = br.readLine().split(":")[1].trim();
			String AccessTokenSecret = br.readLine().split(":")[1].trim();
			authorizationBearer = br.readLine().split(":")[1].trim();

			br = new BufferedReader( new FileReader( new File( "./files/ners.txt") ) );
			String line = null;
			while( ( line = br.readLine() ) != null ){
				ners.put( line.trim(), "TRUE" );
			}

			br = new BufferedReader( new FileReader( new File( "./files/phrases.txt") ) );
			line = null;
			while( ( line = br.readLine() ) != null ){
				phrs.put( line.trim(), "TRUE" );
			}

			br = new BufferedReader( new FileReader( new File( "./files/stopWords.txt") ) );
			while( ( line = br.readLine() ) != null ){
				stopWords.add( line.trim().toLowerCase() );
			}

		}catch(Exception e){
			e.printStackTrace();
		}
	}

	static boolean validNER(String nnp){
		String symbols = ";,@$%6i^&*()_+:'\"";
		for(int i = 0; i < nnp.length(); i++)
			if( symbols.indexOf( nnp.charAt( i ) ) >= 0 )
				return false;
		return true;
	}

	public static String cleanOriginal(String original){
		String[] tokens = original.split("\\s");
		String word = "";
		String symbols = ";.'\",!@$^*-+=<>/?~`|\\#";
		for(String token: tokens){
			if( symbols.indexOf( token.trim() ) >= 0  )
				continue;
			word = word.trim()+" "+token;
		}
		original = word.trim();
	return original;
	}

	public static String cleanKey(String nnp){
		String[] tokens = nnp.split("\\s");
		String symbols = ";.'\",!@$%^*(){}[]-+=<>/?~`|/\\:";
		String key = "";
		for(String token: tokens){
			if( stopWords.contains( token.trim().toLowerCase() ) )
				continue;
			boolean flag = false;
			for(int i = 0; i < token.length(); i++){
				if( symbols.indexOf( token.trim().charAt(i) ) >= 0  ){
					flag = true;
					break;
				}
			}
			if( flag )
				continue;
			key = key.trim()+" "+token;
		}
		key = key.trim();
	return key;
	}

	public static String clean(String NE){
		String[] tokens = NE.split("\\s");
		if( tokens.length == 1){
			String LNE = NE.trim().toLowerCase();		
			if( !stopWords.contains( LNE ) && validNER( LNE ) ){
				NE = Character.toUpperCase( NE.charAt(0) ) +""+ NE.substring(1);
				return NE;
			}
			return null;
		}
		String word = "";
		for(String tok: tokens){
			tok = Character.toUpperCase( tok.charAt(0) )  + tok.substring(1);
			if( word.length() == 0)
				word = tok;
			else
				word = word +" "+tok;	
		}

		return word;
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

	void addNNP(String nnp, String type,Long likeCount, Long retweetCount ){

		String key = nnp;
		if( type.equals("NER") ){
			key = cleanKey( nnp );
			if( key == null || key.trim().length() == 0)
			return;
		}
		
		TreeMap<String, Original> entityCount = NNPCount.get( type );

		if( entityCount == null )
			entityCount  = new TreeMap<String, Original>();

		Original count = entityCount.get( key.toUpperCase() );

		if( count == null )
			count = new Original( nnp, likeCount, retweetCount );
		else
			count.add( nnp );	

		NEREntry entry = new NEREntry( key.toUpperCase(), count );

		if( type.equals("NER") && toppers.contains( entry ) ){

			toppers.remove( entry );

			count.count++;
			count.likeCount += likeCount;
			count.retweetCount += retweetCount;	

			entry = new NEREntry( key.toUpperCase(), count );
			toppers.add( entry );
		}else{
			count.count++;
			count.likeCount += likeCount;
			count.retweetCount += retweetCount;	
			
			if( type.equals("NER") ){
				entry = new NEREntry( key.toUpperCase(), count );
				toppers.add( entry );
			}
		}

		entityCount.put( key.toUpperCase(), count );
		NNPCount.put( type,  entityCount);
	}

	void print( PriorityQueue<NEREntry> toppers, int top, int noOfTweets){

		print(3, 0," ___________________________________________________________________");

		String formated = String.format("| #No of Tweets: %-6d              |  count   |  Like   | Retweet |", noOfTweets);
		print(4, 0, formated );

		print(5, 0,"|____________________________________|__________|_________|_________|");

		Queue<NEREntry> copyQueue = new PriorityQueue<NEREntry>( toppers );
    		Iterator<NEREntry> itr = copyQueue.iterator();

		int count = 6;
		int rank = 0;

    		while( itr.hasNext()){
			rank++;
        		NEREntry entry = copyQueue.poll();
			String value = entry.value.getOriginal().trim();
			if( value.length() > 30 ){
				value = value.substring(0,29);
			}
			formated = String.format("| %2d. %-30s | %7d  | %7d | %7d |                                     ", rank, value, entry.value.count, entry.value.likeCount, entry.value.retweetCount );
			print( count, 0, formated); 
			if( rank == top )
				break;
			count++;
		}
		print( count + 1, 0,"|____________________________________|__________|_________|_________|");
	}

	void print( Map<String, TreeMap<String, Original>> NCount, int length, boolean flag){
		System.out.println();
		for(String entityType: NCount.keySet() ){
			Map<String, Original> sortedMap =  SortMapByValue.sortByComparatorOriginal( NCount.get( entityType ) , false);
			int count = 0;
			for(String entity: sortedMap.keySet() ){
				System.out.println( entityType+", "+sortedMap.get( entity ).getOriginal()+", "+ sortedMap.get( entity ).count );
				//sortedMap.get( entity ).print();
				count++;
				if( count ==  length)
					break;
			}
		}
	}

	void print( Map<String, TreeMap<String, Integer>> NCount, int length ){
		System.out.println();
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

	void getRuntimeReplies(String query, int length, String originalKey){

		initCoreNLP();
		clearScreen();

		query = query.replaceAll("\\s", "%20");

		String[] cmd = new String[] {"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query="+query+"&tweet.fields=in_reply_to_user_id,author_id,geo,created_at,public_metrics,conversation_id", "--header", "Authorization:Bearer "+authorizationBearer};
		String[] ocmd = null;

		if( query.indexOf("conversation_id") >= 0){
			ocmd = new String[] { "curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets?ids="+originalKey+"&tweet.fields=attachments,author_id,context_annotations,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,source,text,withheld&expansions=referenced_tweets.id", "--header", "Authorization: Bearer "+authorizationBearer};
		}
		

		int count = 0;
		while( true ){
			try{
				if( count == 0 && ocmd != null ){

					ProcessBuilder builder = new ProcessBuilder( ocmd );
					Process process = builder.start();
					BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

					String mainTweet = "";
					String line = null;
					while ((line = reader.readLine()) != null) {
						mainTweet += line;
					}

					process.waitFor();
					JSONObject obj =  ( JSONObject ) ( new JSONParser().parse( mainTweet ) );
					JSONArray arr = (JSONArray) obj.get("data");
					Iterator itr = arr.iterator();

					while ( itr.hasNext() ) {

						JSONObject jo = (JSONObject) itr.next();
						String tweet = (String ) jo.get("text");
						JSONObject pm = ( JSONObject ) jo.get("public_metrics");
						Long likes  = (Long)  pm.get("like_count");
						Long retweet  = (Long)  pm.get("retweet_count");
						Long reply  = (Long)  pm.get("reply_count");

						String formated1 = String.format("Tweet: %-60s",tweet);
						String formated2 = String.format("Replies:%-4d, Likes:%-4d, Retweets:%-4d",likes,retweet,reply);

						print(1,0, formated1);
						print(2,0, formated2);
					}
				}else if( count == 0){
					print(1,0, "Search For: "+originalKey);
				}

				
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

					JSONObject jo = (JSONObject) itr.next();
					String tweet = (String ) jo.get("text");
					JSONObject pm = ( JSONObject ) jo.get("public_metrics");

					Long likeCount  = (Long)  pm.get("like_count");
					Long retweetCount  = (Long)  pm.get("retweet_count");

					CoreDocument doc = new CoreDocument( tweet );
					pipeline.annotate( doc );

					String NNP = "";
					String PR = "";
					String tag = "";
					String original = "";
					count++;

					for (CoreLabel tok : doc.tokens()) {

						if( tag.trim().length() > 0 ){
							original = tag;
							tag = tag+"|"+tok.tag().trim();
						}else{ 
							tag = tok.tag().trim();
							original = tag;
						}
					
						if( tok.word().trim().indexOf("http:") >= 0 || tok.word().trim().indexOf("https:") >= 0  ){
							addNER( expand ( tok.word().trim() ), "URL", likeCount, retweetCount );
						}else if( tok.word().trim().charAt(0) == '@' ){
							addNER( tok.word().trim(), "TWITTER", likeCount, retweetCount );
						}else if( ners.containsKey( tag ) ){
							NNP = NNP.trim()+" "+tok.word().trim();
						}else {
							addNER( NNP.trim(), "NER" , likeCount, retweetCount );
							tag = ""; NNP = "";
						}
					}
					count++;
					addNER( NNP.trim(),"NER", likeCount, retweetCount );
					print( toppers, 10, count );
				}
				cmd = new String[]{"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query="+query+"&tweet.fields=in_reply_to_user_id,author_id,created_at,geo,conversation_id,public_metrics&next_token="+nToken, "--header", "Authorization:Bearer "+authorizationBearer};

				//cmd = new String[]{"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query="+query+"&tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id&next_token="+nToken, "--header", "Authorization:Bearer "+authorizationBearer};
			}catch(Exception e){
				break;
			}
		}
		print( NNPCount, length, true );

	}

	List<String> getReplies(String query, int size){

		List<String> reply = new ArrayList<String>();
		String[] cmd = new String[] {"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query="+query+"&tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id", "--header", "Authorization:Bearer "+authorizationBearer};

		/*
		   for(String c: cmd)
		   System.out.print(c+" ");
		   System.out.println();
		 */

		int count = 0;
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

				cmd = new String[]{"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query="+query+"&tweet.fields=in_reply_to_user_id,author_id,created_at,geo,conversation_id,public_metrics&next_token="+nToken, "--header", "Authorization:Bearer "+authorizationBearer};
			}catch(Exception e){
				break;
			}
		}
		System.out.println();
		return reply;
	}		

	boolean validNNP(String nnp){
		String symbols = ";,$%6i^&*()_+:'\"";
		if( symbols.indexOf( nnp ) >= 0 )
			return false;
		return true;
	}

	void addNER(String NNP, String type, Long likeCount, Long retweetCount ){

		if( NNP.length() >  0 ) {
			if( type.equals("NER") || type.equals("PHRASE")){	
				NNP = clean( NNP );
			}
			if( NNP != null )
				addNNP( NNP, type, likeCount, retweetCount);
		}
	}

	void analyser(List<String> replies, int length, boolean isNER){

		initCoreNLP();
		clearScreen();
		double count = 0.0d;
		int noOfTweets = 0;
		for(String reply: replies){
			CoreDocument doc = new CoreDocument( reply );
			pipeline.annotate( doc );
			count++;
			double done = ( count / (double) replies.size() ) * 100;
			String strDouble = String.format("%.2f", done);
			print(1,0, "Processing:("+count+" Doc) "+strDouble+"% Completed");
			if( isNER ){
				for (CoreEntityMention em : doc.entityMentions()){
					addNE( em );
				}
			}
			String NNP = "";
			String PR = "";
			String pTag = "";
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

				/*

				   if( pTag.trim().length() > 0){
				   pTag = pTag+"|"+tok.tag().trim();
				   }else{
				   pTag = tag;
				   }

				 */

				if( tok.word().trim().indexOf("http:") >= 0 || tok.word().trim().indexOf("https:") >= 0  ){
					addNER( expand ( tok.word().trim() ), "URL" , 0L, 0L);
					//tag = ""; NNP = ""; pTag = ""; PR = "";
				}else if( tok.word().trim().charAt(0) == '@' ){
					addNER( tok.word().trim(), "TWITTER", 0L, 0L );
					//tag = ""; NNP = ""; pTag = ""; PR = "";

				}else if( ners.containsKey( tag ) ){
					NNP = NNP.trim()+" "+tok.word().trim();
				}else {
					addNER( NNP.trim(), "NER" , 0L, 0L);
					tag = ""; NNP = "";
				}

				/*

				   if( phrs.containsKey( pTag ) ){
				   PR = PR.trim()+" "+tok.word().trim();
				   } else {
				   addNER( PR.trim(), "PHRASE");
				   pTag = ""; PR = "";
				   }
				 */	
			}
			addNER( NNP.trim(),"NER" , 0L, 0L);
			noOfTweets++;
			if( count % 2 == 0 ){
				print( toppers, 10,  noOfTweets );
			}
			//addNER( PR.trim(),"PHRASE" );
			/*
			   for (CoreSentence sent : doc.sentences()) {
			   System.err.println(sent.text());
			   }

			   System.out.println("tokens and ner ners");
			   String tokensAndNERTags = doc.tokens().stream().map(token -> "("+token.word()+","+token.ner()+")").collect( Collectors.joining(" "));
			   System.err.println(tokensAndNERTags);
			 */
		}
		if( isNER )
			print( NECount, length );

		clearScreen();	
		System.out.println("--------------------------------");
		print( NNPCount, length, true );
	}

	void clearScreen(){
		System.out.print("\033[H\033[2J");  
		System.out.flush();  		
	}

	void print(int row,int column, String test){
		char escCode = 0x1B;
		System.out.print(String.format("%c[%d;%df\b%s",escCode,row,column, test));
	}

	public static void main(String[] args) throws Throwable {

		Replies reply = new Replies();
		String conversation_id = null;
		String search = null;
		int i = 0;
		int size = -1;
		int length = 20;
		boolean isNER = false;
		boolean runtime = false;

		while( i < args.length ){
			if( args[i].equals("-ci") || args[i].equals("-c") )
				conversation_id = args[ ++i ];
			if( args[i].equals("-se") || args[i].equals("-s") )
				search = args[ ++i ];
			if( args[i].equals("-sz") )
				size = Integer.parseInt( args[ ++i ] );
			if( args[i].equals("-ln") )
				length = Integer.parseInt( args[ ++i ] );
			if( args[i].equals("-ne") )
				isNER = true;
			if( args[i].equals("-r") ||  args[i].equals("-run") )
				runtime = true;
			i++;
		}

		if( ( conversation_id == null && search == null ) || size ==  -1){
			System.err.println( "-ci <conversation_id> -se <search-key> -sz <size> -ln <length> -ne [NER] -r/-run [runtime]");
			System.exit(0);
		}

		String query = "";
		String key = "";

		if( conversation_id != null ){
			query = "conversation_id:"+conversation_id;
			key = conversation_id;
		}else if( search != null ){
			query = search;
			key = search;
		}

		if( runtime ){
			reply.getRuntimeReplies( query, length, key );
		}else{
			reply.analyser(  reply.getReplies( query, size ), length, isNER );
		}

		System.out.println("-------------- END -------------------");
	}
}

