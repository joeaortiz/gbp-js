# Visualising Gaussian Belief Propagation

### 1D GBP

Hosted [here](https://joeaortiz.github.io/gbp-distill/components/1dgbp.html). 

To host locally:

```
git clone git@github.com:joeaortiz/gbp-distill.git
cd gbp-distill

brew install npm
npm init -y
npm install ml-matrix --save
npm install browserify --save-dev
npx browserify js/1dgbp.js -o js/bundles/1dgbp.js

python -m http.server
```
Open in a browser:
```angular2
http://localhost:8000/components/1dgbp.html
```
