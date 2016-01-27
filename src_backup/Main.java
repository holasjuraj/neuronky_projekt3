import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

public class Main {

	public static final String TRAIN_IMAGE = "data/angelina.jpg";
	public static final String[] TEST_IMAGES = {"data/angelina.jpg", "data/emma.jpg", "data/lenna.jpg"};
	public static final int BLOCK_HEIGHT = 10;
	public static final int BLOCK_WIDTH = 10;
	public static final int COMPONENTS = 8;
	
	private static int imageHeightBlocks = 1;
	
	public static void main(String[] args) {
//		testAPEX("data/run");
		testMLP("data/runMLP2");
	}
	
	public static void testAPEX(String outputDir){
//		for(int r = 0; r < 30; r++){ final String OUTPUT_DIR = "data/set3/run"+(r+1); System.out.println("\nRUN "+(r+1)+":");
		// Training
		List<double[]> dataSet = readData(TRAIN_IMAGE);
		APEX apex = new APEX(BLOCK_HEIGHT*BLOCK_WIDTH, COMPONENTS, 0.001);
//		apex.analyticPCA(dataSet);
		double[] errs = apex.train(dataSet, 1000);
		
		// Principal components and eigenvalues
		File outDir = new File(outputDir);
		if(!outDir.exists()){
			outDir.mkdir();
		}
		saveData(apex.getPComponents(), outputDir+"/components.png", 1);
		double[] eig = apex.getEigValues();
		try {
			PrintWriter pw = new PrintWriter(new File(outputDir+"/eigenvalues.txt"));
			for(int i = 0; i < eig.length; i++){
				pw.println(eig[i]);
				System.out.println(eig[i]);
			}
			pw.close();
		}
		catch (FileNotFoundException e) { e.printStackTrace(); }
		
		// Images reconstruction
		for(String image : TEST_IMAGES){
			dataSet = readData(image);
			for(int p = 1; p <= COMPONENTS; p++){
				saveData(apex.reconstruct(dataSet, p),
						outputDir+"/reconstructed_"+image.substring(image.lastIndexOf("/")+1, image.indexOf("."))+p+".png",
						imageHeightBlocks);
			}
			System.out.println(image + " completed");
		}
		
		// Errors
		for(int i = 0; i < errs.length; i++){
			System.out.println(errs[i]);
		}
//		}		
	}
	
	public static void testMLP(String outputDir){
		List<double[]> dataSet = readData(TRAIN_IMAGE);
		
		File outDir = new File(outputDir);
		if(!outDir.exists()){
			outDir.mkdir();
		}
		
		for(int p = COMPONENTS; p <= COMPONENTS; p++){
			// Training
			MLP mlp = new MLP(BLOCK_HEIGHT*BLOCK_WIDTH, p, 0.001);
			double[] errs = mlp.train(dataSet, 1000);
			// Images reconstruction
			for(String image : TEST_IMAGES){
				List<double[]> dataTest = readData(image);
				saveData(mlp.reconstruct(dataTest),
						outputDir+"/reconstructed_"+image.substring(image.lastIndexOf("/")+1, image.indexOf("."))+p+".png",
						imageHeightBlocks);
				System.out.println(p + " components, " + image + " completed");
			}
			// Errors
			for(int i = 0; i < errs.length; i++){
				System.out.println(errs[i]);
			}
		}
		
	}
	
	public static List<double[]> readData(String path){
		// Open image
		File imgPath = new File(path);
		BufferedImage bi = null;
		try {
			// Transform to TYPE_BYTE_GRAY
			BufferedImage orig = ImageIO.read(imgPath);
			bi = new BufferedImage(orig.getWidth(), orig.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
			bi.getGraphics().drawImage(orig, 0, 0, null);
		}
		catch (IOException e) { e.printStackTrace(); }
		WritableRaster raster = bi.getRaster();
		DataBufferByte dbb = (DataBufferByte) raster.getDataBuffer();
		byte[] bytes = dbb.getData();
		
		// Split to blocks
		int numBlocks = bytes.length / (BLOCK_HEIGHT*BLOCK_WIDTH),
			widthBlocks = bi.getWidth() / BLOCK_WIDTH;
		imageHeightBlocks = bi.getHeight() / BLOCK_HEIGHT;
		List<double[]> dataSet = new ArrayList<double[]>(numBlocks);
		for(int b = 0; b < numBlocks; b++){
			double[] data = new double[BLOCK_HEIGHT*BLOCK_WIDTH];
			int bx = b % widthBlocks;
			int by = b / widthBlocks;
			for(int y = 0; y < BLOCK_HEIGHT; y++){
				for(int x = 0; x < BLOCK_WIDTH; x++){
					byte col = bytes[(by*BLOCK_HEIGHT + y)*bi.getWidth()  +  (bx*BLOCK_WIDTH + x)];
					data[y*BLOCK_WIDTH + x] = (double)((col+256) % 256);					
				}
			}
			dataSet.add(data);
		}
		return dataSet;
	}
	
	public static void saveData(List<double[]> dataSet, String outputPath, int heightBlocks){
		// Merge blocks
		int dataLength = dataSet.get(0).length,
			numBlocks = dataSet.size(),
			widthBlocks = numBlocks / heightBlocks,
			width = widthBlocks * BLOCK_WIDTH,
			height = heightBlocks * BLOCK_HEIGHT;
		int[] bytes = new int[dataSet.size() * dataLength];

		for(int b = 0; b < numBlocks; b++){
			double[] data = dataSet.get(b);
			int bx = b % widthBlocks;
			int by = b / widthBlocks;
			for(int y = 0; y < BLOCK_HEIGHT; y++){
				for(int x = 0; x < BLOCK_WIDTH; x++){
					long col = Math.round(data[y*BLOCK_WIDTH + x]);
					col = Math.max(col, 0);
					col = Math.min(col, 255);
					bytes[(by*BLOCK_HEIGHT + y)*width  +  (bx*BLOCK_WIDTH + x)] = (byte)col;
				}
			}
		}

		// Write image
		try {
			BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
			WritableRaster raster = img.getRaster();
			raster.setPixels(0, 0, width, height, bytes);
			ImageIO.write(img, "PNG", new File(outputPath));
		}
		catch (Exception e) { e.printStackTrace(); }
	}

}
