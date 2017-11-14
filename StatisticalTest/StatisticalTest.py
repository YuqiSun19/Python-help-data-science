from scipy import stats
import numpy as np 

#--------analysis on one sample--------#
np.random.seed(282629734)
x=stats.t.rvs(10, size=1000) #t-distribution with degree of freedom 10

print ("maximum, minimum, mean, variance:", x.max(), x.min(), x.mean(), x.var())

#theoretical: 
tm, tv, ts, tk=stats.t.stats(10, moments='mvsk')
#sample:
sn, (smin, smax), sm, sv, ss, sk=stats.describe(x)
print ("theoretical distribution:")
sstr = 'mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'
print (sstr % (tm, tv, ts, tk))
print ("samples:")
print (sstr % (sm, sv, ss, sk))

#t-test: test hypothes whether sample mean is the same as expectation   
print ('t-statistic= %6.3f pvalue=%6.4f' % stats.ttest_1samp(x, tm))
#p-value is large, cannot reject sample mean

#Kolmogorov-Smirnov test: test hypothesis whether sample comes from standard t-distribution
print ('KS-statistic D=%6.3f pvalue=%6.3f' % stats.kstest(x, 't', (10, )) )
#p-value is large, cannot reject hypothesis

#Kolmogorov-Smirnov test: test hypothesis whether sample comes from standard normal distribution
print ('KS-statistic D=%6.3f pvalue=%6.3f' % stats.kstest(x, 'norm') )
#p-value is large, cannot reject hypothesis

#normaltest: test hypothesis whether sample comes from normal distribution
print ('normal test D=%6.3f pvalue=%6.3f' % stats.normaltest(x))
#p-value is low, can reject hypothesis

#-----Example-----#
#--generate 10 normal distributional random variables with mean=3, and sigma^2=5
#--based on 95% confidence interval, compute sample mean, and sample mean with error
#--replicate 100 times
#--compute real percentage contained in confidence interval
def case(n = 10, mu = 3, sigma = np.sqrt(5), p = 0.025, rep = 100):
    m = np.zeros((rep, 4))

    for i in range(rep):
      norm = np.random.normal(loc = mu, scale = sigma, size = n)
      xbar = np.mean(norm)
      low = xbar - ss.norm.ppf(q = 1 - p) * (sigma / np.sqrt(n))
      up = xbar + ss.norm.ppf(q = 1 - p) * (sigma / np.sqrt(n))

      if (mu > low) & (mu < up):
          rem = 1
      else:
          rem = 0

      m[i,:] = [xbar, low, up, rem]
    inside = np.sum(m[:,3])
    per = inside / rep
    desc = "There are " + str(inside) + " confidence intervals that contain the true mean (" + str(mu) + "), that is " + str(per) + " percent of the total CIs"

    return {"Matrix": m, "Decision": desc}
