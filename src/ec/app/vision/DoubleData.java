/*
  Copyright 2006 by Sean Luke
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/


package ec.app.vision;
import ec.gp.*;

public class DoubleData extends GPData
    {
    /**
	 * 
	 */
	private static final long serialVersionUID = -372980601815138802L;
	public double x;    // return value

    public void copyTo(final GPData gpd)   // copy my stuff to another DoubleData
        { ((DoubleData)gpd).x = x; }
    }


