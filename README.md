# Visualising Gaussian Belief Propagation

## Usage
```
git clone git@github.com:joeaortiz/gbp-distill.git
cd gbp-distill
git checkout gh-pages
```

### 1D Surface Estimation

```
open components/1dgbp.html  # or gnome-open
```
Alternatively, hosted [here](https://joeaortiz.github.io/gbp-distill/components/1dgbp.html).

### 2D Robot Simulation

```
open components/2dgbp.html  # or gnome-open
```
Alternatively, hosted [here](https://joeaortiz.github.io/gbp-distill/components/2dgbp.html).

## Development

```bash
brew install npm

# install dependencies
npm install

# build JS bundles
npx browserify js/1dgbp.js -o js/bundles/1dgbp.js
npx browserify js/2dgbp.js -o js/bundles/2dgbp.js
```
