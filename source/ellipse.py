import pandas as pd # import panda package, and we call it pd. (saves the data as a data frame fromat)
import matplotlib.pyplot as plt # this package will be use to draw some usful graphs (like bar graphs)
import numpy as np 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy
from sklearn.metrics import mean_squared_error, r2_score

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs): # https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
def get_coorelationCoffe (x, y):
     return (np.corrcoef( [np.array(x).flatten(), np.array(y).flatten()] )[0][1] )
    
def error (coeff, x): # I need some explnation here, is this the SDevation error
    sd = np.sqrt( (1-(coeff**2))/(len(x)) )
    return sd
def p_value (coeff, sd): # I need some explnation here, is this name "p_value" correct ?
    z_score = coeff/sd
    p = scipy.stats.norm.sf(abs(z_score))*2
    return p



def draw_confidence_ellipse (x_c1, y_c1, x_c2, y_c2, xaxis, yaxis, sol, xl, yl):
    fig, ax_nstd = plt.subplots(figsize=(6, 6))


    arr_class = sol.split("vs")
    ax_nstd.scatter(np.array(x_c1).flatten(), np.array(y_c1).flatten(), s=6, color = 'blue')
    confidence_ellipse(np.array(x_c1).flatten(), np.array(y_c1).flatten(), ax_nstd, n_std=2,label=arr_class[0], edgecolor='blue', linestyle='--')


    ax_nstd.scatter(np.array(x_c2).flatten(),np.array(y_c2).flatten(), s=6, color = 'orange')

    confidence_ellipse(np.array(x_c2).flatten(), np.array(y_c2).flatten(), ax_nstd, n_std=2,label=arr_class[1], edgecolor='orange', linestyle='--')


    ax_nstd.set_title(sol,fontsize =16, fontweight ='bold')
    plt.xlabel(xaxis, fontweight ='bold', fontsize =16)
    plt.ylabel(yaxis, fontweight ='bold', fontsize =16)
    plt.ylim(yl)
    plt.xlim(xl)

    ax_nstd.legend(prop={"size":14})
    plt.show()
    #### Drawing the graphs ends here, now with the calculation, here we are just using the functions above. thanks
#     print()
#     print(arr_class[0])
#     fer_coeff = get_coorelationCoffe (x_c1, y_c1)
#     fer_error = error(fer_coeff, x_c1)
#     fer_p = p_value(fer_coeff, fer_error)
    
#     print( "correlation coefficient:" + str(fer_coeff) )
#     print("error: "+str(fer_error))
#     print("p: " +str(fer_p))
    
    
#     print()
#     print(arr_class[1])
#     nf_coeff = get_coorelationCoffe (x_c2, y_c2)
#     nf_error = error(nf_coeff, x_c2)
#     nf_p = p_value(nf_coeff, nf_error)
    
#     print( "correlation coefficient:" + str(nf_coeff) )
#     print("error: "+str(nf_error))
#     print("p: " +str(nf_p))
