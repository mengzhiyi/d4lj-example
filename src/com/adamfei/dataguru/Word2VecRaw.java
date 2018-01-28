package com.adamfei.dataguru;

import java.io.FileNotFoundException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class Word2VecRaw {
	
	public static void testW2d(SentenceIterator iter, TokenizerFactory t){
		Word2Vec vec = new Word2Vec.Builder()
					.allowParallelTokenization(true)
					.minWordFrequency(2)
					.iterations(1)
					.layerSize(100)
					.seed(42)
					.windowSize(5)
					.iterate(iter)
					.tokenizerFactory(t)
					.build();
		
		System.out.println("Fitting Word2Vec Model ....");
		vec.fit();
		
		WordVectorSerializer.writeWord2VecModel(vec, "w2vm.txt");
		System.out.println("和stock最接近的10个词汇:\n\t" + vec.wordsNearest("stock", 10));
		System.out.println("和school最接近的10个词汇:\n\t" + vec.wordsNearest("school", 10));
		
		
	}
	
	public static void testP2v(SentenceIterator iter, TokenizerFactory t){
		ParagraphVectors pvec = new ParagraphVectors.Builder()
				.allowParallelTokenization(true)
				.minWordFrequency(1)
				.iterations(1)
				.layerSize(100)
				.seed(42)
				.windowSize(5)
				.iterate(iter)
				.tokenizerFactory(t)
				.build();
		
		System.out.println("Fitting Doc2v model ......");
		pvec.fit();
		
		WordVectorSerializer.writeParagraphVectors(pvec, "d2vm.txt");
		System.err.println("和stock最接近的10个词汇:\n\t" + 
				pvec.wordsNearest("stock", 10) );
		
		System.err.println("和school最接近的10个词汇:\n\t" + 
				pvec.wordsNearest("school", 10) );
	}
	
	public static void run(){
		String filePath = "D:\\JavaWorkSpace\\dl4j-project\\dl4j-examples\\dl4j-examples\\src\\main\\resources\\NewsData\\RawNewsToGenerateWordVector.txt";
	
		try {
			SentenceIterator iter = new BasicLineIterator(filePath);
			TokenizerFactory t = new DefaultTokenizerFactory();
			t.setTokenPreProcessor(new CommonPreprocessor());
			
			System.out.println("Building model .....");
			
			testW2d(iter, t);
			System.out.println("=================================");
			testP2v(iter, t);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		Word2VecRaw.run();
	}
}
