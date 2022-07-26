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

class TagGenerator{

	StanfordCoreNLP pipeline = null;
	Map<String,String> tagList; 

	TagGenerator(){
		tagList = new TreeMap<String, String>();
	}

	void addTag(String tags){
		String[] toks = tags.split("\\|");
		String add = "";	
		for(String tag: toks){
			if(add.length()==0)
				add = tag;
			else
				add = add+"|"+tag;
			System.out.println("Added: "+add);
			tagList.put(add,"true");
		}
	}

	void print(){
		for(String key: tagList.keySet() ){
			System.out.println(key);
		}
	}

	void initCoreNLP(){
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, sentiment");
		pipeline = new StanfordCoreNLP( props );
	}

	void generate(String filename){
		initCoreNLP();
		try{
			BufferedReader br = new BufferedReader( new FileReader( new File(filename) ));		
			String line = null;
			while( ( line = br.readLine() )  != null ){

				if( line.trim().length() >  0){
					CoreDocument doc = new CoreDocument( line );
					pipeline.annotate( doc );
					String tag = "";
					String name = "";
					for (CoreLabel tok : doc.tokens()) {
						if( tag.trim().length() == 0)
							tag = tok.tag();
						else
							tag = tag+"|"+tok.tag();

						if( name.trim().length() == 0 )
							name = tok.word();
						else
							name = name+" "+tok.word();
					}
					addTag( tag.trim() );
					System.out.println(tag.trim()+" = "+name.trim());
				}
			}
			//print();
		}catch(Exception e){
			e.printStackTrace();
		}
	}

	public static void main(String[] args) throws Throwable {
		TagGenerator tg = new TagGenerator();
		//tg.generate("./files/list-of-movies.txt");
		//tg.generate("./files/companies.txt");
		//tg.generate("./files/names.txt");
		tg.generate("./files/videogames.txt");
		tg.print();
	}
}

