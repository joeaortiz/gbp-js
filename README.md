# Visualising Gaussian Belief Propagation

## Usage
```
git clone git@github.com:joeaortiz/gbp-distill.git
cd gbp-distill
git checkout gh-pages
```

### 1D GBP

```
open components/1dgbp.html  # or gnome-open
```

## Development

```bash
brew install npm

# install dependencies
npm install

# build JS bundles
npx browserify js/1dgbp.js -o js/bundles/1dgbp.js
```
