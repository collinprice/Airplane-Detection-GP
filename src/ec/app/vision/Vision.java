package ec.app.vision;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import ec.util.*;
import ec.*;
import ec.app.vision.DoubleData;
import ec.gp.*;
import ec.gp.koza.*;
import ec.simple.*;

/**********************************************************************
 * 
 * Collin Price
 * cp06vz @ brocku.ca
 * 
 * Apr., 15, 2011
 * 
 **********************************************************************/

public class Vision extends GPProblem implements SimpleProblemForm
    {
    /**
	 * 
	 */
	private static final long serialVersionUID = 8886023049823102982L;
	
	
	private static BufferedImage[] 	trainingData;
	private static boolean[][] 		trainingDataMask;
	private static int[][]			trainingDataPoints;
	private static boolean[][] 		testingDataMask;
	private static int[][]			testingDataPoints;
	
	private static double[][]		stdDeviation;
	private static double[][]		meanVals;
	private static int				total_true;
	private static int				total_false;
	
	
	private static BufferedImage[] testingData;
	
	public int currentp1;
	public int currentp2;
	public int currentp3;
	public int currentp4;
	public int currentp5;
	public int currentp6;
	public int currentp7;
	public int currentp8;
	public int currentp9;
	public double stdDev;
	public double mean;

    public Object clone() {
        Vision newobj = (Vision) (super.clone());
        return newobj;
    }

    /**
     * Setup the GP for training.
     */
    public void setup(final EvolutionState state, final Parameter base) {
        super.setup(state,base);
        System.out.println("Running setup...");
        int trainingSize = state.parameters.getInt(base.push("training.size"), null, 1);
        int testingSize = state.parameters.getInt(base.push("testing.size"), null, 1);
        
        trainingData = new BufferedImage[trainingSize];
        BufferedImage[] dataMask = new BufferedImage[trainingSize];
        BufferedImage[] testingMask = new BufferedImage[testingSize];
        testingData = new BufferedImage[testingSize];
        
        // Reading Training Data
        for (int i = 0; i < trainingSize; i++) {
        	try {
        		trainingData[i] = ImageIO.read(state.parameters.getFile(base.push("training." + i + ".image"), null));
        		dataMask[i] = ImageIO.read(state.parameters.getFile(base.push("training." + i + ".image.mask"), null));
			} catch (IOException e) {
				System.out.println("Cannot find an image...");
				e.printStackTrace();
				System.exit(1);
			}
        }
        // reading testing data
        for (int i = 0; i < testingSize; i++) {
        	try {
        		testingData[i] = ImageIO.read(state.parameters.getFile(base.push("testing." + i + ".image"), null));
        		testingMask[i] = ImageIO.read(state.parameters.getFile(base.push("testing." + i + ".image.mask"), null));
			} catch (IOException e) {
				System.out.println("Cannot find an image...");
				e.printStackTrace();
				System.exit(1);
			}
        }
        
		// caches training data for faster runs
        trainingDataPoints = new int[trainingData[0].getWidth()][trainingData[0].getHeight()];
        for (int i = 0; i < trainingData[0].getWidth(); i++) {
        	for (int j = 0; j < trainingData[0].getHeight(); j++) {
        		Color pixel = new Color(trainingData[0].getRGB(i, j));
        		trainingDataPoints[i][j] = pixel.getBlue();
        	}
        }
        
		// caches testing data for faster runs
        testingDataPoints = new int[testingData[0].getWidth()][testingData[0].getHeight()];
        for (int i = 0; i < testingData[0].getWidth(); i++) {
        	for (int j = 0; j < testingData[0].getHeight(); j++) {
        		Color pixel = new Color(testingData[0].getRGB(i, j));
        		testingDataPoints[i][j] = pixel.getBlue();
        	}
        }
        
        
        trainingDataMask = new boolean[dataMask[0].getWidth()][dataMask[0].getHeight()];
        total_true = 0;
        total_false = 0;
        // parse training masks
        for (int i = 0; i < dataMask[0].getWidth(); i++) {
        	for (int j = 0; j < dataMask[0].getHeight(); j++) {
        		Color pixel = new Color(dataMask[0].getRGB(i, j));
        		
        		if (pixel.getRed() == pixel.getBlue()) {
        			trainingDataMask[i][j] = false;
        			total_false++;
        		} else {
        			trainingDataMask[i][j] = true;
        			total_true++;
        		}
        	}
        }
        
        testingDataMask = new boolean[testingMask[0].getWidth()][testingMask[0].getHeight()];
        // parse testing masks
        for (int i = 0; i < testingMask[0].getWidth(); i++) {
        	for (int j = 0; j < testingMask[0].getHeight(); j++) {
        		Color pixel = new Color(testingMask[0].getRGB(i, j));
        		
        		if (pixel.getRed() == pixel.getBlue()) {
        			testingDataMask[i][j] = false;
        		} else {
        			testingDataMask[i][j] = true;
        		}
        	}
        }
        
        calcStdDev(trainingDataPoints);
    } // setup

    /**
     * Evaluation for each individual in the population.
     */
    public void evaluate(final EvolutionState state, final Individual ind, 
        final int subpopulation, final int threadnum) {
    	
        if (!ind.evaluated) {
        	int TP = 0; // true positive
        	int FP = 0; // false positive
        	int TN = 0; // true negative
        	int FN = 0; // false negative
          	
            	BufferedImage img = trainingData[0];
            	for (int x = 1; x < img.getWidth()-1; x++) {
            		for (int y = 1; y < img.getHeight()-1; y++) {
            		
            			
            			DoubleData input = new DoubleData();
            			
            			currentp1 = trainingDataPoints[x-1][y-1];
            			currentp2 = trainingDataPoints[x][y-1];
            			currentp3 = trainingDataPoints[x+1][y-1];
            			currentp4 = trainingDataPoints[x-1][y];
            			currentp5 = trainingDataPoints[x][y];
            			currentp6 = trainingDataPoints[x+1][y];
            			currentp7 = trainingDataPoints[x-1][y+1];
            			currentp8 = trainingDataPoints[x][y+1];
            			currentp9 = trainingDataPoints[x+1][y+1];
            			stdDev = stdDeviation[x][y];
            			mean = meanVals[x][y];
            			
            			((GPIndividual)ind).trees[0].child.eval(
                                state,threadnum,input,stack,((GPIndividual)ind),this);
            			
            			if (input.x > 0) {
            				if (trainingDataMask[x][y]) {
            					TP++;
            				} else {
            					FP++;
            				}
            			} else {
            				if (!trainingDataMask[x][y]) {
            					TN++;
            				} else {
            					FN++;
            				}
            			}
            		}
            	}

            int fitness = FP + FN;
            
            KozaFitness f = ((KozaFitness)ind.fitness);
            f.setStandardizedFitness(state, fitness);
            f.hits = TP + TN;
            ind.evaluated = true;
        }
    }
    
	/*
		This method is run at the end of the GP experiment. It runs through
		all of the testing images and prints them out.
	*/
    public void closeContacts(EvolutionState state, int results) {
    	calcStdDev(testingDataPoints);
    	BufferedImage coded = null;
    	BufferedImage new_mask = null;
    	
    	SimpleStatistics stats = (SimpleStatistics) state.statistics;
    	Individual ind = stats.best_of_run[0];
        /* Test each image in the training set. */
        for (int m = 0; m < testingData.length; m++) {
        	int TP = 0;
        	int TN = 0;
        	int FP = 0;
        	int FN = 0;
        	BufferedImage img = testingData[m];
        	
        	coded = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        	new_mask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        	for (int y = 1; y < img.getHeight()-1; y++) {
        		for (int x = 1; x < img.getWidth()-1; x++) {
        		
        			DoubleData input = new DoubleData();
        			
        			currentp1 = testingDataPoints[x-1][y-1];
        			currentp2 = testingDataPoints[x][y-1];
        			currentp3 = testingDataPoints[x+1][y-1];
        			currentp4 = testingDataPoints[x-1][y];
        			currentp5 = testingDataPoints[x][y];
        			currentp6 = testingDataPoints[x+1][y];
        			currentp7 = testingDataPoints[x-1][y+1];
        			currentp8 = testingDataPoints[x][y+1];
        			currentp9 = testingDataPoints[x+1][y+1];
        			stdDev = stdDeviation[x][y];
        			mean = meanVals[x][y];
        			
        			
        			((GPIndividual)ind).trees[0].child.eval(
                            state,0,input,stack,((GPIndividual)ind),this);
        			
        			if (input.x >= 0) {
        				coded.setRGB(x, y, Color.red.getRGB());
        				if (testingDataMask[x][y]) {
        					new_mask.setRGB(x, y, Color.green.getRGB());
        					TP++;
        				} else {
        					new_mask.setRGB(x, y, Color.yellow.getRGB());
        					FP++;
        				}
        			} else {
        				coded.setRGB(x, y, new Color(testingDataPoints[x][y],testingDataPoints[x][y],testingDataPoints[x][y]).getRGB());
        				if (!testingDataMask[x][y]) {
        					new_mask.setRGB(x, y, Color.blue.getRGB());
        					TN++;
        				} else {
        					new_mask.setRGB(x, y, Color.red.getRGB());
        					FN++;
        				}
        			}
        		}
        	}
        	state.output.println("", stats.statisticslog);
        	state.output.println("Testing TP = " + TP + ", TN = " + TN ,stats.statisticslog);
        	state.output.println("Testing FP = " + FP + ", FN = " + FN ,stats.statisticslog);
        	short name = (short)System.currentTimeMillis();
        	try {
				ImageIO.write(coded, "png", new File("output" + name + "_masked" + ".png"));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			try {
				ImageIO.write(new_mask, "png", new File("output" + name + ".png"));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        }
    } //closeContacts
    
    /*
     * Calculates the Standard Deviation and Mean of each pixel.
     */
    private void calcStdDev(int[][] pop) {
    	stdDeviation = new double[pop.length][pop[0].length];
    	meanVals = new double[pop.length][pop[0].length];
    	
    	for (int i = 1; i < pop.length-1; i++) {
    		for (int j = 1; j < pop[0].length-1; j++) {
    			int total = 0;
    			
    			int[] subset = {
    					pop[i-1][j-1],
    					pop[i][j-1],
    					pop[i+1][j-1],
    					pop[i-1][j],
    					pop[i][j],
    					pop[i+1][j],
    					pop[i-1][j+1],
    					pop[i][j+1],
    					pop[i+1][j+1]
    					         };
    			
    			for (int x : subset) {
    				total += x;
    			}
    			
    			int mean = total / subset.length;
    			
    			total = 0;
    			for (int x : subset) {
    				total += x - mean;
    			}
    			
    			stdDeviation[i][j] = Math.sqrt(total / subset.length);
    			meanVals[i][j] = mean;
    		}
    	}
    } // calcStdDev
    
} // Regression

