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
	PriorityQueue<TEntry> priorityQueue;
	Map<String,Integer> counts;

	Original(String original){

		this.original = Replies.cleanOriginal( original );
		this.count = 0;	
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

	void addNNP(String nnp, String type ){
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
			count = new Original( nnp );
		else
			count.add( nnp );	

		NEREntry entry = new NEREntry( key.toUpperCase(), count );

		if( type.equals("NER") && toppers.contains( entry ) ){
			toppers.remove( entry );
			count.count++;
			entry = new NEREntry( key.toUpperCase(), count );
			toppers.add( entry );
		}else{
			count.count++;
			if( type.equals("NER") ){
				entry = new NEREntry( key.toUpperCase(), count );
				toppers.add( entry );
			}
		}

		entityCount.put( key.toUpperCase(), count );
		NNPCount.put( type,  entityCount);
	}

	void print( PriorityQueue<NEREntry> toppers, int top){
		print(2,0, "------------NER------------");
		Queue<NEREntry> copyQueue = new PriorityQueue<NEREntry>( toppers );
    		Iterator<NEREntry> itr = copyQueue.iterator();
		int count = 1;
    		while( itr.hasNext()){
        		NEREntry entry = copyQueue.poll();
			print( 2 + count, 0, count+". "+ entry.value.getOriginal()+"  "+entry.value.count+"                                ");
			if( count == top )
				break;
			count++;
		}
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

				cmd = new String[]{"curl", "--request", "GET", "--url", "https://api.twitter.com/2/tweets/search/recent?query="+query+"&tweet.fields=in_reply_to_user_id,author_id,created_at,conversation_id&next_token="+nToken, "--header", "Authorization:Bearer "+authorizationBearer};
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

	void addNER(String NNP, String type ){

		if( NNP.length() >  0 ) {
			if( type.equals("NER") || type.equals("PHRASE")){	
				NNP = clean( NNP );
			}
			if( NNP != null )
				addNNP( NNP, type );
		}
	}

	void analyser(List<String> replies, int length, boolean isNER){

		initCoreNLP();
		clearScreen();
		double count = 0.0d;

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
					addNER( expand ( tok.word().trim() ), "URL" );
					//tag = ""; NNP = ""; pTag = ""; PR = "";
				}else if( tok.word().trim().charAt(0) == '@' ){
					addNER( tok.word().trim(), "TWITTER" );
					//tag = ""; NNP = ""; pTag = ""; PR = "";

				}else if( ners.containsKey( tag ) ){
					NNP = NNP.trim()+" "+tok.word().trim();
				}else {
					addNER( NNP.trim(), "NER" );
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
			addNER( NNP.trim(),"NER" );
			if( count % 2 == 0 ){
				print( toppers, 10 );
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
			i++;
		}

		if( ( conversation_id == null && search == null ) || size ==  -1){
			System.err.println( "-ci <conversation_id> -se <search-key> -sz <size> -ln <length> -ne [NER]");
			System.exit(0);
		}

		String query = "";

		if( conversation_id != null ){
			query = "conversation_id:"+conversation_id;
		}else if( search != null ){
			query = search;
		}

		reply.analyser(  reply.getReplies( query, size ), length, isNER );
	}
}

