(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
var m = require('ml-matrix');
var r = require('random');

// ****************************** Gaussian ************************************
class Gaussian {
  constructor(eta, lam) {
    // TODO: Consider saving the dimension
    if ((eta instanceof m.Matrix) && (lam instanceof m.Matrix)) {
      this.eta = eta;
      this.lam = lam;
    } else {
      this.eta = new m.Matrix([eta]).transpose();
      this.lam = new m.Matrix(lam);
    }
  }

  getCov() {
    return m.inverse(this.lam);
  }

  getMean() {
    const cov = this.getCov();
    return cov.mmul(this.eta);
  }

  // Take product of gaussian with other gaussian
  product(gaussian) {
    this.eta.add(gaussian.eta);
    this.lam.add(gaussian.lam);
  }

  getCovEllipse() {
    var cov = this.getCov();
    var e = new m.EigenvalueDecomposition(cov);
    var real = e.realEigenvalues;

    // Eigenvectors are always orthogonal
    var vectors = e.eigenvectorMatrix;
    var angle = Math.atan(vectors.get(1, 0) / vectors.get(0, 0))
    return [real, angle]
  }
}

function getEllipse(cov) {
  var e = new m.EigenvalueDecomposition(cov);
  var real = e.realEigenvalues;

  // Eigenvectors are always orthogonal
  var vectors = e.eigenvectorMatrix;
  var angle = Math.atan(vectors.get(1, 0) / vectors.get(0, 0))
  return [real, angle]
}


// ranges from [i1,i2) & [j1,j2)
function sliceMat(mat,i1,j1,i2,j2) {
  const result = new m.Matrix(i2-i1,j2-j1);
  for(let i = i1; i < i2; ++i) 
    for(let j = j1; j < j2; ++j) {
      result.set(i-i1, j-j1, mat.get(i,j)); 
    }
  return result;
}


// ****************************** GBP ************************************

class FactorGraph {
  constructor() {
    this.pose_nodes = [];
    this.lmk_nodes = [];

    this.factors = [];
  }

  update_beliefs() {
    for(var c=0; c<this.pose_nodes.length; c++) {
      this.pose_nodes[c].update_belief();
    }
    for(var c=0; c<this.lmk_nodes.length; c++) {
      this.lmk_nodes[c].update_belief();
    }
 
  }

  send_messages() {
    for(var c=0; c<this.factors.length; c++) {
      this.factors[c].send_mess();
    }

  }

  sync_iter() {
    this.send_messages();
    this.update_beliefs();
  }

  computeMAP() {
    var tot_dofs = 0;
    for(var c=0; c<this.pose_nodes.length; c++) {
      tot_dofs += this.pose_nodes[c].dofs;
    }
    for(var c=0; c<this.lmk_nodes.length; c++) {
      tot_dofs += this.lmk_nodes[c].dofs;
    }

    const bigEta = m.Matrix.zeros(tot_dofs, 1);
    const bigLam = m.Matrix.zeros(tot_dofs, tot_dofs);

    // Add priors
    var l_dofs = 2 * this.lmk_nodes.length;
    for(var c=0; c<this.pose_nodes.length; c++) {
      new m.MatrixSubView(bigEta, l_dofs+c*2, l_dofs+(c+1)*2-1, 0, 0).add(this.pose_nodes[c].prior.eta);
      new m.MatrixSubView(bigLam, l_dofs+c*2, l_dofs+(c+1)*2-1, l_dofs+c*2, l_dofs+(c+1)*2-1).add(this.pose_nodes[c].prior.lam);
    }
    for(var c=0; c<this.lmk_nodes.length; c++) {
      new m.MatrixSubView(bigEta, c*2, (c+1)*2-1, 0, 0).add(this.lmk_nodes[c].prior.eta);
      new m.MatrixSubView(bigLam, c*2, (c+1)*2-1, c*2, (c+1)*2-1).add(this.lmk_nodes[c].prior.lam);
    }

    // Add factors
    for(var c=0; c<this.factors.length; c++) {
      if (this.factors[c].adj_var_ids[1] < n_landmarks) {
        const f_pose_eta = new m.MatrixSubView(this.factors[c].factor.eta, 0, 1, 0, 0);
        const f_lmk_eta = new m.MatrixSubView(this.factors[c].factor.eta, 2, 3, 0, 0);
        const f_pose_lam = new m.MatrixSubView(this.factors[c].factor.lam, 0, 1, 0, 1);
        const f_lmk_lam = new m.MatrixSubView(this.factors[c].factor.lam, 2, 3, 2, 3);
        const f_pose_lmk_lam = new m.MatrixSubView(this.factors[c].factor.lam, 0, 1, 2, 3);
        const f_lmk_pose_lam = new m.MatrixSubView(this.factors[c].factor.lam, 2, 3, 0, 1);

        var c_id = graph.lmk_nodes.length + this.factors[c].adj_var_ids[0] - n_landmarks;
        var l_id = lmk_graph_ix[this.factors[c].adj_var_ids[1]];
        new m.MatrixSubView(bigEta, c_id*2, (c_id+1)*2-1, 0, 0).add(f_pose_eta);
        new m.MatrixSubView(bigEta, l_id*2, (l_id+1)*2-1, 0, 0).add(f_lmk_eta);
        new m.MatrixSubView(bigLam, c_id*2, (c_id+1)*2-1, c_id*2, (c_id+1)*2-1).add(f_pose_lam);
        new m.MatrixSubView(bigLam, l_id*2, (l_id+1)*2-1, l_id*2, (l_id+1)*2-1).add(f_lmk_lam);
        new m.MatrixSubView(bigLam, c_id*2, (c_id+1)*2-1, l_id*2, (l_id+1)*2-1).add(f_pose_lmk_lam);
        new m.MatrixSubView(bigLam, l_id*2, (l_id+1)*2-1, c_id*2, (c_id+1)*2-1).add(f_lmk_pose_lam);
      } else {
        const f_p1_eta = new m.MatrixSubView(this.factors[c].factor.eta, 0, 1, 0, 0);
        const f_p2_eta = new m.MatrixSubView(this.factors[c].factor.eta, 2, 3, 0, 0);
        const f_p1_lam = new m.MatrixSubView(this.factors[c].factor.lam, 0, 1, 0, 1);
        const f_p2_lam = new m.MatrixSubView(this.factors[c].factor.lam, 2, 3, 2, 3);
        const f_p1_p2_lam = new m.MatrixSubView(this.factors[c].factor.lam, 0, 1, 2, 3);
        const f_p2_p1_lam = new m.MatrixSubView(this.factors[c].factor.lam, 2, 3, 0, 1);

        var c_id1 = graph.lmk_nodes.length + this.factors[c].adj_var_ids[0] - n_landmarks;
        var c_id2 = graph.lmk_nodes.length + this.factors[c].adj_var_ids[1] - n_landmarks;
        new m.MatrixSubView(bigEta, c_id1*2, (c_id1+1)*2-1, 0, 0).add(f_p1_eta);
        new m.MatrixSubView(bigEta, c_id2*2, (c_id2+1)*2-1, 0, 0).add(f_p2_eta);
        new m.MatrixSubView(bigLam, c_id1*2, (c_id1+1)*2-1, c_id1*2, (c_id1+1)*2-1).add(f_p1_lam);
        new m.MatrixSubView(bigLam, c_id2*2, (c_id2+1)*2-1, c_id2*2, (c_id2+1)*2-1).add(f_p1_lam);
        new m.MatrixSubView(bigLam, c_id1*2, (c_id1+1)*2-1, c_id2*2, (c_id2+1)*2-1).add(f_p1_p2_lam);
        new m.MatrixSubView(bigLam, c_id2*2, (c_id2+1)*2-1, c_id1*2, (c_id1+1)*2-1).add(f_p2_p1_lam);
      }
    }

    const bigCov = m.inverse(bigLam);
    const means = bigCov.mmul(bigEta);
    return [means, bigCov];
  }

  compare_to_MAP() {
    var gbp_means = [];
    for(var c=0; c<this.lmk_nodes.length; c++) {
      gbp_means.push(this.lmk_nodes[c].belief.getMean().get(0,0));
      gbp_means.push(this.lmk_nodes[c].belief.getMean().get(1,0));
    }
    for(var c=0; c<this.pose_nodes.length; c++) {
      gbp_means.push(this.pose_nodes[c].belief.getMean().get(0,0));
      gbp_means.push(this.pose_nodes[c].belief.getMean().get(1,0));
    }

    const means = new m.Matrix([gbp_means]);
    const map = this.computeMAP()[0];
    var av_diff = (map.sub(means.transpose())).norm();
    return av_diff;
  }

}

class VariableNode {
  constructor(dofs, var_id) {
    this.dofs = dofs;
    this.var_id = var_id;
    this.belief = new Gaussian(m.Matrix.zeros(dofs, 1), m.Matrix.zeros(dofs, dofs));
    this.prior = new Gaussian(m.Matrix.zeros(dofs, 1), m.Matrix.zeros(dofs, dofs));

    this.adj_factors = [];
  }

  update_belief() {
    this.belief.eta = this.prior.eta.clone();
    this.belief.lam = this.prior.lam.clone();

    // Take product of incoming messages
    for(var c=0; c<this.adj_factors.length; c++) {
      var ix = this.adj_factors[c].adj_var_ids.indexOf(this.var_id);
      this.belief.product(this.adj_factors[c].messages[ix])
    }

    // Send new belief to adjacent factors
    for(var c=0; c<this.adj_factors.length; c++) {
      var ix = this.adj_factors[c].adj_var_ids.indexOf(this.var_id);
      this.adj_factors[c].adj_beliefs[ix] = this.belief;
    }
  }
}


class LinearFactor {
  constructor(dofs, adj_var_ids) {
    this.dofs = dofs;
    this.adj_var_ids = adj_var_ids;
    this.adj_beliefs = [];
    this.adj_var_dofs = [];

    // To compute factor when factor is combination of many factor types (e.g. measurement and smoothness)
    this.jacs = [];
    this.meas = [];
    this.lambdas = [];
    this.factor = new Gaussian(m.Matrix.zeros(dofs, 1), m.Matrix.zeros(dofs, dofs));

    this.messages = [];
  }

  compute_factor() {
    this.factor.eta = m.Matrix.zeros(this.dofs, 1);
    this.factor.lam = m.Matrix.zeros(this.dofs, this.dofs);
    for (var i=0; i<this.jacs.length; i++) {
      this.factor.eta.add(this.jacs[i].transpose().mmul(this.meas[i]).mul(this.lambdas[i]));
      this.factor.lam.add(this.jacs[i].transpose().mmul(this.jacs[i]).mul(this.lambdas[i]));
    }
  }

  send_mess() {
    var start_dim = 0;
    
    for (var i=0; i<this.adj_var_ids.length; i++) {
      var eta_factor = this.factor.eta.clone();
      var lam_factor = this.factor.lam.clone();

      // Take product with incoming messages, general for factor connected to arbitrary num var nodes
      var mess_start_dim = 0;
      for (var j=0; j<this.adj_var_ids.length; j++) {
        if (!(i == j)) {
          const eta_prod = m.Matrix.sub(this.adj_beliefs[j].eta, this.messages[j].eta);
          const lam_prod = m.Matrix.sub(this.adj_beliefs[j].lam, this.messages[j].lam);
          new m.MatrixSubView(eta_factor, mess_start_dim, mess_start_dim + this.adj_var_dofs[j] -1, 0, 0).add(eta_prod);
          new m.MatrixSubView(lam_factor, mess_start_dim, mess_start_dim + this.adj_var_dofs[j] -1, mess_start_dim, mess_start_dim + this.adj_var_dofs[j] -1).add(lam_prod);
        }
        mess_start_dim += this.adj_var_dofs[j];
      }

      // For factor connecting 2 variable nodes
      if (i == 0) {
        var eo = new m.MatrixSubView(eta_factor, 0, 1, 0, 0);
        var eno = new m.MatrixSubView(eta_factor, 2, 3, 0, 0);
        var loo = new m.MatrixSubView(lam_factor, 0, 1, 0, 1);
        var lnono = new m.MatrixSubView(lam_factor, 2, 3, 2, 3);
        var lnoo = new m.MatrixSubView(lam_factor, 2, 3, 0, 1);
        var lono = new m.MatrixSubView(lam_factor, 0, 1, 2, 3);
      } else if (i == 1) {
        var eno = new m.MatrixSubView(eta_factor, 0, 1, 0, 0);
        var eo = new m.MatrixSubView(eta_factor, 2, 3, 0, 0);
        var lnono = new m.MatrixSubView(lam_factor, 0, 1, 0, 1);
        var loo = new m.MatrixSubView(lam_factor, 2, 3, 2, 3);
        var lono = new m.MatrixSubView(lam_factor, 2, 3, 0, 1);
        var lnoo = new m.MatrixSubView(lam_factor, 0, 1, 2, 3);
      }

      const mess = new Gaussian([[0],[0]], [[0,0],[0,0]]);
      const block = lono.mmul(m.inverse(lnono));
      mess.eta = new m.Matrix(eo.sub(block.mmul(eno)));
      mess.lam = new m.Matrix(loo.sub(block.mmul(lnoo)));
      this.messages[i] = mess;
    }

  }
}


// ***************************** Graph management functions ****************************

function syncGBP() {
  graph.sync_iter();
  if (!(n_iters == 0)) {
    dist = graph.compare_to_MAP();   
  }
  n_iters++;
}

function get_dist_from_gt() {
  // distance from gt
  var lmk_dist_from_gt = 0;
  var pose_dist_from_gt = 0;
  for(var j=0; j<graph.lmk_nodes.length; j++) {
    lmk_dist_from_gt += Math.sqrt(Math.pow(landmarks_gt[j].x - graph.lmk_nodes[j].belief.getMean().get(0, 0), 2) + 
                         Math.pow(landmarks_gt[j].y - graph.lmk_nodes[j].belief.getMean().get(1, 0), 2));
  }
  for(var j=0; j<graph.pose_nodes.length; j++) {
   pose_dist_from_gt += Math.sqrt(Math.pow(poses_gt[j].x - graph.pose_nodes[j].belief.getMean().get(0, 0), 2) + 
                         Math.pow(poses_gt[j].y - graph.pose_nodes[j].belief.getMean().get(1, 0), 2));
  }
  lmk_dist_from_gt /= graph.lmk_nodes.length;
  pose_dist_from_gt /= graph.pose_nodes.length;

  return [pose_dist_from_gt, lmk_dist_from_gt]  
}

function addLandmarkNode(ix) {
  const lmk_node = new VariableNode(2, ix);
  var lambda = 1 / Math.pow(lmk_prior_std, 2);
  lmk_node.prior.lam = new m.Matrix([[lambda, 0], [0, lambda]]);
  lmk_node.prior.eta = lmk_node.prior.lam.mmul(new m.Matrix([[landmarks_gt[ix].x], [landmarks_gt[ix].y]]));
  lmk_node.update_belief();
  lmk_graph_ix[ix] = graph.lmk_nodes.length;
  graph.lmk_nodes.push(lmk_node);
}

// Add odometry factor connecting to most recent pose to penultimate pose
function addOdometryFactor() {
  var n_pose_nodes = graph.pose_nodes.length;

  const odometry_factor = new LinearFactor(4, [graph.pose_nodes[n_pose_nodes-2].var_id, graph.pose_nodes[n_pose_nodes-1].var_id]);
  odometry_factor.jacs.push(meas_jac);
  const measurement = new m.Matrix([[poses_gt[n_pose_nodes-1].x - poses_gt[n_pose_nodes-2].x + odometry_noise()], 
    [poses_gt[n_pose_nodes-1].y - poses_gt[n_pose_nodes-2].y  + odometry_noise()]])
  odometry_factor.meas.push(measurement);
  odometry_factor.lambdas.push(1 / Math.pow(odometry_std, 2));
  odometry_factor.adj_var_dofs.push(2);
  odometry_factor.adj_var_dofs.push(2);

  odometry_factor.adj_beliefs.push(graph.pose_nodes[n_pose_nodes-2].belief);
  odometry_factor.adj_beliefs.push(graph.pose_nodes[n_pose_nodes-1].belief);

  odometry_factor.messages.push(new Gaussian([[0],[0]], [[0,0],[0,0]]));
  odometry_factor.messages.push(new Gaussian([[0],[0]], [[0,0],[0,0]]));
  odometry_factor.compute_factor();
  graph.factors.push(odometry_factor);

  graph.pose_nodes[n_pose_nodes-2].adj_factors.push(odometry_factor);
  graph.pose_nodes[n_pose_nodes-1].adj_factors.push(odometry_factor);
}

function addMeasurementFactors() {
  // Add measurement factors connecting to observed landmarks
  var n_pose_nodes = graph.pose_nodes.length;
  for (var j=0; j<n_landmarks; j++) {
    var dist = Math.sqrt(Math.pow(landmarks_gt[j].x - poses_gt[n_pose_nodes -1].x, 2) + 
                       Math.pow(landmarks_gt[j].y - poses_gt[n_pose_nodes -1].y, 2));
    if (dist < meas_range) {
      // Create new landmark node if first observation of the landmark
      if (!(lmk_observed_yet[j])) {
        addLandmarkNode(j);
        lmk_observed_yet[j] = 1;
      }

      const new_factor = new LinearFactor(4, [graph.pose_nodes[n_pose_nodes-1].var_id, j]);
      new_factor.jacs.push(meas_jac);
      const measurement = new m.Matrix([[landmarks_gt[j].x - poses_gt[n_pose_nodes-1].x + measurement_noise()], 
        [landmarks_gt[j].y - poses_gt[n_pose_nodes-1].y + measurement_noise()]])
      new_factor.meas.push(measurement);
      new_factor.lambdas.push(1 / Math.pow(meas_std, 2));
      new_factor.adj_var_dofs.push(2);
      new_factor.adj_var_dofs.push(2);

      new_factor.adj_beliefs.push(graph.pose_nodes[n_pose_nodes-1].belief);
      new_factor.adj_beliefs.push(graph.lmk_nodes[lmk_graph_ix[j]].belief);

      new_factor.messages.push(new Gaussian([[0],[0]], [[0,0],[0,0]]));
      new_factor.messages.push(new Gaussian([[0],[0]], [[0,0],[0,0]]));
      new_factor.compute_factor();
      graph.factors.push(new_factor);

      graph.pose_nodes[n_pose_nodes-1].adj_factors.push(new_factor);
      graph.lmk_nodes[lmk_graph_ix[j]].adj_factors.push(new_factor);
    }
  }
}

function checkAddVarNode() {
  var dist = Math.sqrt(Math.pow(robot_loc[0] - last_key_pose[0], 2) + 
                       Math.pow(robot_loc[1] - last_key_pose[1], 2));

  if (dist > new_pose_dist) {
    const new_var_node = new VariableNode(2, n_landmarks + graph.pose_nodes.length);
    var lambda = 1 / Math.pow(robot_prior_std, 2);
    new_var_node.prior.lam = new m.Matrix([[lambda, 0], [0, lambda]]);
    new_var_node.prior.eta = new_var_node.prior.lam.mmul(new m.Matrix([[robot_loc[0]], [robot_loc[1]]]))
    new_var_node.update_belief();
    graph.pose_nodes.push(new_var_node);
    poses_gt.push({x: robot_loc[0], y: robot_loc[1]});
    last_key_pose[0] = robot_loc[0];
    last_key_pose[1] = robot_loc[1];

    addOdometryFactor();
    addMeasurementFactors();
  }
}


// ******************************* Drawing functions ********************************

function drawCanvasBackground() {
  ctx.fillStyle = "grey";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawRobot() {
  ctx.beginPath();
  ctx.arc(robot_loc[0], robot_loc[1], var_node_radius, 0, Math.PI*2);
  ctx.fillStyle = "red";
  ctx.fill();
  ctx.closePath();
}

function drawPoseNodes() {
  for(var c=0; c<graph.pose_nodes.length; c++) {
    const mean = graph.pose_nodes[c].belief.getMean();
    var x = mean.get(0, 0);
    var y = mean.get(1, 0);

    // Draw means
    ctx.beginPath();
    ctx.arc(x, y, var_node_radius, 0, Math.PI*2);
    ctx.fillStyle = "#0095DD";
    ctx.fill();
    ctx.closePath();

    var values = graph.pose_nodes[c].belief.getCovEllipse();
    var eig_values = values[0];
    var angle = values[1];

    // Draw variances
    ctx.beginPath();
    ctx.ellipse(x, y, Math.sqrt(eig_values[0]), Math.sqrt(eig_values[1]), angle, 0, 2*Math.PI)
    ctx.strokeStyle = "#0095DD";
    ctx.stroke();
  }
}

function drawLandmarkNodes() {
  for(var c=0; c<graph.lmk_nodes.length; c++) {
    const mean = graph.lmk_nodes[c].belief.getMean();
    var x = mean.get(0, 0);
    var y = mean.get(1, 0);

    // Draw means
    ctx.beginPath();
    ctx.arc(x, y, var_node_radius, 0, Math.PI*2);
    ctx.fillStyle = "yellow";
    ctx.fill();
    ctx.closePath();

    var values = graph.lmk_nodes[c].belief.getCovEllipse();
    var eig_values = values[0];
    var angle = values[1];

    // Draw variances
    ctx.beginPath();
    ctx.ellipse(x, y, Math.sqrt(eig_values[0]), Math.sqrt(eig_values[1]), angle, 0, 2*Math.PI)
    ctx.strokeStyle = "yellow";
    ctx.stroke();
  }
}

function drawPosesGT() {
  for(var i=0; i<poses_gt.length; i++) {
    ctx.beginPath();
    ctx.arc(poses_gt[i].x, poses_gt[i].y, gt_node_radius, 0, Math.PI*2);
    ctx.fillStyle = "black";
    ctx.fill();
    ctx.closePath();
  }
}

function drawLandmarksGT() {
  for (var i=0; i<landmarks_gt.length; i++) {
    ctx.beginPath();
    ctx.arc(landmarks_gt[i].x, landmarks_gt[i].y, gt_node_radius, 0, Math.PI*2);
    ctx.fillStyle = "orange";
    ctx.fill();
    ctx.closePath();
  }
}

function drawLines() {
  for (var c=0; c<graph.factors.length; c++) {
    ctx.beginPath();
    if ((graph.factors[c].adj_var_ids[1] < n_landmarks)) {
      const mean0 = graph.pose_nodes[graph.factors[c].adj_var_ids[0] - n_landmarks].belief.getMean();
      const mean1 = graph.lmk_nodes[lmk_graph_ix[graph.factors[c].adj_var_ids[1]]].belief.getMean();
      ctx.moveTo(mean0.get(0,0), mean0.get(1,0));
      ctx.lineTo(mean1.get(0,0), mean1.get(1,0));
      ctx.strokeStyle = "black";
    } else {
      const mean0 = graph.pose_nodes[graph.factors[c].adj_var_ids[0] - n_landmarks].belief.getMean();
      const mean1 = graph.pose_nodes[graph.factors[c].adj_var_ids[1] - n_landmarks].belief.getMean();
      ctx.moveTo(mean0.get(0,0), mean0.get(1,0));
      ctx.lineTo(mean1.get(0,0), mean1.get(1,0)); 
      ctx.strokeStyle = "blue";
    }
      ctx.stroke();
  }
}

function drawMAP() {
  var values = graph.computeMAP();
  const means = values[0];
  const bigSigma = values[1];
  for(var c=0; c<graph.lmk_nodes.length; c++) {
    const mean = new m.MatrixSubView(means, c*2, c*2+1, 0, 0);
    const Sigma = new m.MatrixSubView(bigSigma, c*2, c*2+1, c*2, c*2+1);
    var x = mean.get(0, 0);
    var y = mean.get(1, 0);

    // Draw means
    ctx.beginPath();
    ctx.arc(x, y, var_node_radius, 0, Math.PI*2);
    ctx.fillStyle = "green";
    ctx.fill();
    ctx.closePath();

    var values = getEllipse(Sigma);
    var eig_values = values[0];
    var angle = values[1];

    // Draw variances
    ctx.beginPath();
    ctx.ellipse(x, y, Math.sqrt(eig_values[0]), Math.sqrt(eig_values[1]), angle, 0, 2*Math.PI)
    ctx.strokeStyle = "green";
    ctx.stroke();
  }
  for(var c=0; c<graph.pose_nodes.length; c++) {
    const i = c + graph.lmk_nodes.length;
    const mean = new m.Matrix(new m.MatrixSubView(means, i*2, i*2+1, 0, 0));
    const Sigma = new m.Matrix(new m.MatrixSubView(bigSigma, i*2, i*2+1, i*2, i*2+1));
    var x = mean.get(0, 0);
    var y = mean.get(1, 0);

    // Draw means
    ctx.beginPath();
    ctx.arc(x, y, var_node_radius, 0, Math.PI*2);
    ctx.fillStyle = "green";
    ctx.fill();
    ctx.closePath();

    var values = getEllipse(Sigma);
    var eig_values = values[0];
    var angle = values[1];

    // Draw variances
    ctx.beginPath();
    ctx.ellipse(x, y, Math.sqrt(eig_values[0]), Math.sqrt(eig_values[1]), angle, 0, 2*Math.PI)
    ctx.strokeStyle = "green";
    ctx.stroke();
  }
}

function drawDistance() {
    ctx.font = "16px Arial";
    ctx.fillStyle = "black";
    ctx.fillText("Av dist from MAP: "+dist.toFixed(4), 8, 20);
}

function drawNumIters() {
    ctx.font = "16px Arial";
    ctx.fillStyle = "black";
    ctx.fillText("Num iterations: "+n_iters, canvas.width - 170, 20);
}


function updateVis() {
  requestAnimationFrame(updateVis);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawCanvasBackground();

  fpsInterval = 1000 / iters_per_sec;
  now = Date.now();
  elapsed = now - then;
  if (elapsed > fpsInterval) {
    then = now - (elapsed % fpsInterval);
    if (GBP_on) {
      syncGBP();
    } 
  }

  if (GBP_on) {
    drawNumIters();
    drawDistance();
  } 
  
  drawLines();
  drawRobot();
  drawPoseNodes();
  drawLandmarkNodes();
  drawPosesGT();
  drawLandmarksGT();
  if (disp_MAP) {
    drawMAP();
  }
}

function start(fps) {
    then = Date.now();
    startTime = then;
    updateVis();
}

// ****************************** User controls *******************************

// Sliders
function update_meas_model_std(val) {
  meas_std = val;
  var lambda = 1 / Math.pow(meas_std, 2);
  // document.getElementById("mmstd").innerHTML = meas_model_std;
  for (var c=0; c<graph.factors.length; c++) {
    if (graph.factors[c].adj_var_ids[1] < n_landmarks) {
      graph.factors[c].lambdas[0] = lambda;
      graph.factors[c].compute_factor();
    }
  }
}

function update_odometry_model_std(val) {
  odometry_std = val;
  var lambda = 1 / Math.pow(odometry_std, 2);
  // document.getElementById("sstd").innerHTML = smoothness_std; 
  for (var c=0; c<graph.factors.length; c++) {
    if (graph.factors[c].adj_var_ids[1] >= n_landmarks) {
      graph.factors[c].lambdas[0] = lambda;
      graph.factors[c].compute_factor();
    }
  }
}

function update_iters_per_sec(val) {
  iters_per_sec = val;
  document.getElementById("fps_2d").innerHTML = iters_per_sec; 
}

var mm_slider = document.getElementById("meas_model_std_2d");
mm_slider.oninput = function() { update_meas_model_std(this.value)}
var s_slider = document.getElementById("odometry_model_std_2d");
s_slider.oninput = function() { update_odometry_model_std(this.value)}
var i_slider = document.getElementById("iters_per_sec_2d");
i_slider.oninput = function() { update_iters_per_sec(this.value)}


// Buttons
function addButton(name, text, func) {
  var buttonnode = document.getElementById(name);
  buttonnode.setAttribute('value', text);
  buttonnode.addEventListener ("click", func, false);
}

addButton('sync_2d', 'Start synchronous GBP', start_syncGBP);
addButton('map_2d', 'Display MAP', display_MAP);

function start_syncGBP() {
  if (graph.factors.length == 0) {
    alert("Move the robot! You must have a factor in the graph to begin GBP.")
  } else {
    if (GBP_on) {
      GBP_on = 0;
      this.value = ("Resume synchronous GBP");
    } else {
      GBP_on = 1;
      iters_per_sec = i_slider.value;
      this.value = ("Pause synchronous GBP");
    }
  }
}
function display_MAP() {
  if (disp_MAP == 0) {
    disp_MAP = 1;
    this.value = ("Hide MAP");
  } else {
    disp_MAP = 0;
    this.value = ("Display MAP");
  }
}

// On click
// document.addEventListener("click", addLandmark, false);
function addLandmark(e) {
  var relativeX = e.clientX - canvas.offsetLeft;
  var relativeY = e.clientY - canvas.offsetTop;
  if(relativeX > 0 && relativeX < canvas.width && relativeY > 0 && relativeY < canvas.height) {
    landmarks.push({x: relativeX, y: relativeY});
  }
}

// On key press
document.addEventListener("keydown", checkKey);
function checkKey(e) {
  e = e || window.event;
  if (e.keyCode == '87') {
    if (robot_loc[1] > var_node_radius + step) {
      robot_loc[1] -= step;
    }
  }
  else if (e.keyCode == '83') {
    if (robot_loc[1] < canvas.height - var_node_radius - step) {
      robot_loc[1] += step;
    }
  }
  else if (e.keyCode == '65') {
    if (robot_loc[0] > var_node_radius + step) {
      robot_loc[0] -= step;
    }
  }
  else if (e.keyCode == '68') {
    if (robot_loc[0] < canvas.width - var_node_radius - step) {
      robot_loc[0] += step;
    }
  }
  checkAddVarNode();
}


// ****************************** Run GBP ************************************

// TO DO
// Make it look cleaner
// Place landmarks around the edge
// Add key

// Visual varaibles
var canvas = document.getElementById("canvas_2d");
var ctx = canvas.getContext("2d");
ctx.lineWidth = 2;

var var_node_radius = 7;
var gt_node_radius = 4;

var disp_MAP = 0;

// Robot motion params
var robot_loc = [50, 590];
var last_key_pose = [50, 590];
var step = 10;
var new_pose_dist = 70;

var n_landmarks = 20;
var landmarks_gt = [];
var poses_gt = [];
var lmk_observed_yet = [];
var lmk_graph_ix = [];

// GBP params
var meas_range = 120;
var lmk_prior_std = 50;
var robot_prior_std = 50;

var GBP_on = 0;
var n_iters = 0;
var dist = 0;
var iters_per_sec = 50;

// Measurement models
const meas_jac = new m.Matrix([[-1., 0., 1., 0.], [0., -1., 0., 1.]]);

const odometry_noise = r.normal(0, 10);
const measurement_noise = r.normal(0, 10);
var odometry_std = 50;
var meas_std = 50;

// Create initial factor graph
const graph = new FactorGraph();
const first_var_node = new VariableNode(2, n_landmarks);
first_var_node.prior.eta = new m.Matrix([[robot_loc[0]], [robot_loc[1]]]);
first_var_node.prior.lam = new m.Matrix([[1, 0], [0, 1]]);  // strong prior for first measurement
first_var_node.update_belief();
graph.pose_nodes.push(first_var_node);
poses_gt.push({x: robot_loc[0], y: robot_loc[1]})

// Generate landmarks
for (var i=0; i<n_landmarks; i++) {
  var x = Math.random()*(canvas.width-20) + 10;
  var y = Math.random()*(canvas.height-20) + 10;
  landmarks_gt.push({x: x, y: y});
  lmk_observed_yet.push(0);
  lmk_graph_ix.push(-1);
}
addMeasurementFactors();  // add initial measurements

start();

},{"ml-matrix":7,"random":30}],2:[function(require,module,exports){

},{}],3:[function(require,module,exports){
'use strict';

const toString = Object.prototype.toString;

function isAnyArray(object) {
  return toString.call(object).endsWith('Array]');
}

module.exports = isAnyArray;

},{}],4:[function(require,module,exports){
'use strict';

function _interopDefault (ex) { return (ex && (typeof ex === 'object') && 'default' in ex) ? ex['default'] : ex; }

var isArray = _interopDefault(require('is-any-array'));

/**
 * Computes the maximum of the given values
 * @param {Array<number>} input
 * @return {number}
 */
function max(input) {
  if (!isArray(input)) {
    throw new TypeError('input must be an array');
  }

  if (input.length === 0) {
    throw new TypeError('input must not be empty');
  }

  let maxValue = input[0];
  for (let i = 1; i < input.length; i++) {
    if (input[i] > maxValue) maxValue = input[i];
  }
  return maxValue;
}

module.exports = max;

},{"is-any-array":3}],5:[function(require,module,exports){
'use strict';

function _interopDefault (ex) { return (ex && (typeof ex === 'object') && 'default' in ex) ? ex['default'] : ex; }

var isArray = _interopDefault(require('is-any-array'));

/**
 * Computes the minimum of the given values
 * @param {Array<number>} input
 * @return {number}
 */
function min(input) {
  if (!isArray(input)) {
    throw new TypeError('input must be an array');
  }

  if (input.length === 0) {
    throw new TypeError('input must not be empty');
  }

  let minValue = input[0];
  for (let i = 1; i < input.length; i++) {
    if (input[i] < minValue) minValue = input[i];
  }
  return minValue;
}

module.exports = min;

},{"is-any-array":3}],6:[function(require,module,exports){
'use strict';

function _interopDefault (ex) { return (ex && (typeof ex === 'object') && 'default' in ex) ? ex['default'] : ex; }

var max = _interopDefault(require('ml-array-max'));
var min = _interopDefault(require('ml-array-min'));
var isArray = _interopDefault(require('is-any-array'));

function rescale(input, options = {}) {
  if (!isArray(input)) {
    throw new TypeError('input must be an array');
  } else if (input.length === 0) {
    throw new TypeError('input must not be empty');
  }

  let output;
  if (options.output !== undefined) {
    if (!isArray(options.output)) {
      throw new TypeError('output option must be an array if specified');
    }
    output = options.output;
  } else {
    output = new Array(input.length);
  }

  const currentMin = min(input);
  const currentMax = max(input);

  if (currentMin === currentMax) {
    throw new RangeError(
      'minimum and maximum input values are equal. Cannot rescale a constant array',
    );
  }

  const {
    min: minValue = options.autoMinMax ? currentMin : 0,
    max: maxValue = options.autoMinMax ? currentMax : 1,
  } = options;

  if (minValue >= maxValue) {
    throw new RangeError('min option must be smaller than max option');
  }

  const factor = (maxValue - minValue) / (currentMax - currentMin);
  for (let i = 0; i < input.length; i++) {
    output[i] = (input[i] - currentMin) * factor + minValue;
  }

  return output;
}

module.exports = rescale;

},{"is-any-array":3,"ml-array-max":4,"ml-array-min":5}],7:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function _interopDefault (ex) { return (ex && (typeof ex === 'object') && 'default' in ex) ? ex['default'] : ex; }

var rescale = _interopDefault(require('ml-array-rescale'));

/**
 * @private
 * Check that a row index is not out of bounds
 * @param {Matrix} matrix
 * @param {number} index
 * @param {boolean} [outer]
 */
function checkRowIndex(matrix, index, outer) {
  let max = outer ? matrix.rows : matrix.rows - 1;
  if (index < 0 || index > max) {
    throw new RangeError('Row index out of range');
  }
}

/**
 * @private
 * Check that a column index is not out of bounds
 * @param {Matrix} matrix
 * @param {number} index
 * @param {boolean} [outer]
 */
function checkColumnIndex(matrix, index, outer) {
  let max = outer ? matrix.columns : matrix.columns - 1;
  if (index < 0 || index > max) {
    throw new RangeError('Column index out of range');
  }
}

/**
 * @private
 * Check that the provided vector is an array with the right length
 * @param {Matrix} matrix
 * @param {Array|Matrix} vector
 * @return {Array}
 * @throws {RangeError}
 */
function checkRowVector(matrix, vector) {
  if (vector.to1DArray) {
    vector = vector.to1DArray();
  }
  if (vector.length !== matrix.columns) {
    throw new RangeError(
      'vector size must be the same as the number of columns',
    );
  }
  return vector;
}

/**
 * @private
 * Check that the provided vector is an array with the right length
 * @param {Matrix} matrix
 * @param {Array|Matrix} vector
 * @return {Array}
 * @throws {RangeError}
 */
function checkColumnVector(matrix, vector) {
  if (vector.to1DArray) {
    vector = vector.to1DArray();
  }
  if (vector.length !== matrix.rows) {
    throw new RangeError('vector size must be the same as the number of rows');
  }
  return vector;
}

function checkIndices(matrix, rowIndices, columnIndices) {
  return {
    row: checkRowIndices(matrix, rowIndices),
    column: checkColumnIndices(matrix, columnIndices),
  };
}

function checkRowIndices(matrix, rowIndices) {
  if (typeof rowIndices !== 'object') {
    throw new TypeError('unexpected type for row indices');
  }

  let rowOut = rowIndices.some((r) => {
    return r < 0 || r >= matrix.rows;
  });

  if (rowOut) {
    throw new RangeError('row indices are out of range');
  }

  if (!Array.isArray(rowIndices)) rowIndices = Array.from(rowIndices);

  return rowIndices;
}

function checkColumnIndices(matrix, columnIndices) {
  if (typeof columnIndices !== 'object') {
    throw new TypeError('unexpected type for column indices');
  }

  let columnOut = columnIndices.some((c) => {
    return c < 0 || c >= matrix.columns;
  });

  if (columnOut) {
    throw new RangeError('column indices are out of range');
  }
  if (!Array.isArray(columnIndices)) columnIndices = Array.from(columnIndices);

  return columnIndices;
}

function checkRange(matrix, startRow, endRow, startColumn, endColumn) {
  if (arguments.length !== 5) {
    throw new RangeError('expected 4 arguments');
  }
  checkNumber('startRow', startRow);
  checkNumber('endRow', endRow);
  checkNumber('startColumn', startColumn);
  checkNumber('endColumn', endColumn);
  if (
    startRow > endRow ||
    startColumn > endColumn ||
    startRow < 0 ||
    startRow >= matrix.rows ||
    endRow < 0 ||
    endRow >= matrix.rows ||
    startColumn < 0 ||
    startColumn >= matrix.columns ||
    endColumn < 0 ||
    endColumn >= matrix.columns
  ) {
    throw new RangeError('Submatrix indices are out of range');
  }
}

function newArray(length, value = 0) {
  let array = [];
  for (let i = 0; i < length; i++) {
    array.push(value);
  }
  return array;
}

function checkNumber(name, value) {
  if (typeof value !== 'number') {
    throw new TypeError(`${name} must be a number`);
  }
}

function sumByRow(matrix) {
  let sum = newArray(matrix.rows);
  for (let i = 0; i < matrix.rows; ++i) {
    for (let j = 0; j < matrix.columns; ++j) {
      sum[i] += matrix.get(i, j);
    }
  }
  return sum;
}

function sumByColumn(matrix) {
  let sum = newArray(matrix.columns);
  for (let i = 0; i < matrix.rows; ++i) {
    for (let j = 0; j < matrix.columns; ++j) {
      sum[j] += matrix.get(i, j);
    }
  }
  return sum;
}

function sumAll(matrix) {
  let v = 0;
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      v += matrix.get(i, j);
    }
  }
  return v;
}

function productByRow(matrix) {
  let sum = newArray(matrix.rows, 1);
  for (let i = 0; i < matrix.rows; ++i) {
    for (let j = 0; j < matrix.columns; ++j) {
      sum[i] *= matrix.get(i, j);
    }
  }
  return sum;
}

function productByColumn(matrix) {
  let sum = newArray(matrix.columns, 1);
  for (let i = 0; i < matrix.rows; ++i) {
    for (let j = 0; j < matrix.columns; ++j) {
      sum[j] *= matrix.get(i, j);
    }
  }
  return sum;
}

function productAll(matrix) {
  let v = 1;
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      v *= matrix.get(i, j);
    }
  }
  return v;
}

function varianceByRow(matrix, unbiased, mean) {
  const rows = matrix.rows;
  const cols = matrix.columns;
  const variance = [];

  for (let i = 0; i < rows; i++) {
    let sum1 = 0;
    let sum2 = 0;
    let x = 0;
    for (let j = 0; j < cols; j++) {
      x = matrix.get(i, j) - mean[i];
      sum1 += x;
      sum2 += x * x;
    }
    if (unbiased) {
      variance.push((sum2 - (sum1 * sum1) / cols) / (cols - 1));
    } else {
      variance.push((sum2 - (sum1 * sum1) / cols) / cols);
    }
  }
  return variance;
}

function varianceByColumn(matrix, unbiased, mean) {
  const rows = matrix.rows;
  const cols = matrix.columns;
  const variance = [];

  for (let j = 0; j < cols; j++) {
    let sum1 = 0;
    let sum2 = 0;
    let x = 0;
    for (let i = 0; i < rows; i++) {
      x = matrix.get(i, j) - mean[j];
      sum1 += x;
      sum2 += x * x;
    }
    if (unbiased) {
      variance.push((sum2 - (sum1 * sum1) / rows) / (rows - 1));
    } else {
      variance.push((sum2 - (sum1 * sum1) / rows) / rows);
    }
  }
  return variance;
}

function varianceAll(matrix, unbiased, mean) {
  const rows = matrix.rows;
  const cols = matrix.columns;
  const size = rows * cols;

  let sum1 = 0;
  let sum2 = 0;
  let x = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      x = matrix.get(i, j) - mean;
      sum1 += x;
      sum2 += x * x;
    }
  }
  if (unbiased) {
    return (sum2 - (sum1 * sum1) / size) / (size - 1);
  } else {
    return (sum2 - (sum1 * sum1) / size) / size;
  }
}

function centerByRow(matrix, mean) {
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) - mean[i]);
    }
  }
}

function centerByColumn(matrix, mean) {
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) - mean[j]);
    }
  }
}

function centerAll(matrix, mean) {
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) - mean);
    }
  }
}

function getScaleByRow(matrix) {
  const scale = [];
  for (let i = 0; i < matrix.rows; i++) {
    let sum = 0;
    for (let j = 0; j < matrix.columns; j++) {
      sum += Math.pow(matrix.get(i, j), 2) / (matrix.columns - 1);
    }
    scale.push(Math.sqrt(sum));
  }
  return scale;
}

function scaleByRow(matrix, scale) {
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) / scale[i]);
    }
  }
}

function getScaleByColumn(matrix) {
  const scale = [];
  for (let j = 0; j < matrix.columns; j++) {
    let sum = 0;
    for (let i = 0; i < matrix.rows; i++) {
      sum += Math.pow(matrix.get(i, j), 2) / (matrix.rows - 1);
    }
    scale.push(Math.sqrt(sum));
  }
  return scale;
}

function scaleByColumn(matrix, scale) {
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) / scale[j]);
    }
  }
}

function getScaleAll(matrix) {
  const divider = matrix.size - 1;
  let sum = 0;
  for (let j = 0; j < matrix.columns; j++) {
    for (let i = 0; i < matrix.rows; i++) {
      sum += Math.pow(matrix.get(i, j), 2) / divider;
    }
  }
  return Math.sqrt(sum);
}

function scaleAll(matrix, scale) {
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) / scale);
    }
  }
}

function inspectMatrix() {
  const indent = ' '.repeat(2);
  const indentData = ' '.repeat(4);
  return `${this.constructor.name} {
${indent}[
${indentData}${inspectData(this, indentData)}
${indent}]
${indent}rows: ${this.rows}
${indent}columns: ${this.columns}
}`;
}

const maxRows = 15;
const maxColumns = 10;
const maxNumSize = 8;

function inspectData(matrix, indent) {
  const { rows, columns } = matrix;
  const maxI = Math.min(rows, maxRows);
  const maxJ = Math.min(columns, maxColumns);
  const result = [];
  for (let i = 0; i < maxI; i++) {
    let line = [];
    for (let j = 0; j < maxJ; j++) {
      line.push(formatNumber(matrix.get(i, j)));
    }
    result.push(`${line.join(' ')}`);
  }
  if (maxJ !== columns) {
    result[result.length - 1] += ` ... ${columns - maxColumns} more columns`;
  }
  if (maxI !== rows) {
    result.push(`... ${rows - maxRows} more rows`);
  }
  return result.join(`\n${indent}`);
}

function formatNumber(num) {
  const numStr = String(num);
  if (numStr.length <= maxNumSize) {
    return numStr.padEnd(maxNumSize, ' ');
  }
  const precise = num.toPrecision(maxNumSize - 2);
  if (precise.length <= maxNumSize) {
    return precise;
  }
  const exponential = num.toExponential(maxNumSize - 2);
  const eIndex = exponential.indexOf('e');
  const e = exponential.substring(eIndex);
  return exponential.substring(0, maxNumSize - e.length) + e;
}

function installMathOperations(AbstractMatrix, Matrix) {
  AbstractMatrix.prototype.add = function add(value) {
    if (typeof value === 'number') return this.addS(value);
    return this.addM(value);
  };

  AbstractMatrix.prototype.addS = function addS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) + value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.addM = function addM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) + matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.add = function add(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.add(value);
  };

  AbstractMatrix.prototype.sub = function sub(value) {
    if (typeof value === 'number') return this.subS(value);
    return this.subM(value);
  };

  AbstractMatrix.prototype.subS = function subS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) - value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.subM = function subM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) - matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.sub = function sub(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.sub(value);
  };
  AbstractMatrix.prototype.subtract = AbstractMatrix.prototype.sub;
  AbstractMatrix.prototype.subtractS = AbstractMatrix.prototype.subS;
  AbstractMatrix.prototype.subtractM = AbstractMatrix.prototype.subM;
  AbstractMatrix.subtract = AbstractMatrix.sub;

  AbstractMatrix.prototype.mul = function mul(value) {
    if (typeof value === 'number') return this.mulS(value);
    return this.mulM(value);
  };

  AbstractMatrix.prototype.mulS = function mulS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) * value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.mulM = function mulM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) * matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.mul = function mul(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.mul(value);
  };
  AbstractMatrix.prototype.multiply = AbstractMatrix.prototype.mul;
  AbstractMatrix.prototype.multiplyS = AbstractMatrix.prototype.mulS;
  AbstractMatrix.prototype.multiplyM = AbstractMatrix.prototype.mulM;
  AbstractMatrix.multiply = AbstractMatrix.mul;

  AbstractMatrix.prototype.div = function div(value) {
    if (typeof value === 'number') return this.divS(value);
    return this.divM(value);
  };

  AbstractMatrix.prototype.divS = function divS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) / value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.divM = function divM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) / matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.div = function div(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.div(value);
  };
  AbstractMatrix.prototype.divide = AbstractMatrix.prototype.div;
  AbstractMatrix.prototype.divideS = AbstractMatrix.prototype.divS;
  AbstractMatrix.prototype.divideM = AbstractMatrix.prototype.divM;
  AbstractMatrix.divide = AbstractMatrix.div;

  AbstractMatrix.prototype.mod = function mod(value) {
    if (typeof value === 'number') return this.modS(value);
    return this.modM(value);
  };

  AbstractMatrix.prototype.modS = function modS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) % value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.modM = function modM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) % matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.mod = function mod(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.mod(value);
  };
  AbstractMatrix.prototype.modulus = AbstractMatrix.prototype.mod;
  AbstractMatrix.prototype.modulusS = AbstractMatrix.prototype.modS;
  AbstractMatrix.prototype.modulusM = AbstractMatrix.prototype.modM;
  AbstractMatrix.modulus = AbstractMatrix.mod;

  AbstractMatrix.prototype.and = function and(value) {
    if (typeof value === 'number') return this.andS(value);
    return this.andM(value);
  };

  AbstractMatrix.prototype.andS = function andS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) & value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.andM = function andM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) & matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.and = function and(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.and(value);
  };

  AbstractMatrix.prototype.or = function or(value) {
    if (typeof value === 'number') return this.orS(value);
    return this.orM(value);
  };

  AbstractMatrix.prototype.orS = function orS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) | value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.orM = function orM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) | matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.or = function or(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.or(value);
  };

  AbstractMatrix.prototype.xor = function xor(value) {
    if (typeof value === 'number') return this.xorS(value);
    return this.xorM(value);
  };

  AbstractMatrix.prototype.xorS = function xorS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) ^ value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.xorM = function xorM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) ^ matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.xor = function xor(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.xor(value);
  };

  AbstractMatrix.prototype.leftShift = function leftShift(value) {
    if (typeof value === 'number') return this.leftShiftS(value);
    return this.leftShiftM(value);
  };

  AbstractMatrix.prototype.leftShiftS = function leftShiftS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) << value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.leftShiftM = function leftShiftM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) << matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.leftShift = function leftShift(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.leftShift(value);
  };

  AbstractMatrix.prototype.signPropagatingRightShift = function signPropagatingRightShift(value) {
    if (typeof value === 'number') return this.signPropagatingRightShiftS(value);
    return this.signPropagatingRightShiftM(value);
  };

  AbstractMatrix.prototype.signPropagatingRightShiftS = function signPropagatingRightShiftS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) >> value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.signPropagatingRightShiftM = function signPropagatingRightShiftM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) >> matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.signPropagatingRightShift = function signPropagatingRightShift(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.signPropagatingRightShift(value);
  };

  AbstractMatrix.prototype.rightShift = function rightShift(value) {
    if (typeof value === 'number') return this.rightShiftS(value);
    return this.rightShiftM(value);
  };

  AbstractMatrix.prototype.rightShiftS = function rightShiftS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) >>> value);
      }
    }
    return this;
  };

  AbstractMatrix.prototype.rightShiftM = function rightShiftM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) >>> matrix.get(i, j));
      }
    }
    return this;
  };

  AbstractMatrix.rightShift = function rightShift(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.rightShift(value);
  };
  AbstractMatrix.prototype.zeroFillRightShift = AbstractMatrix.prototype.rightShift;
  AbstractMatrix.prototype.zeroFillRightShiftS = AbstractMatrix.prototype.rightShiftS;
  AbstractMatrix.prototype.zeroFillRightShiftM = AbstractMatrix.prototype.rightShiftM;
  AbstractMatrix.zeroFillRightShift = AbstractMatrix.rightShift;

  AbstractMatrix.prototype.not = function not() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, ~(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.not = function not(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.not();
  };

  AbstractMatrix.prototype.abs = function abs() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.abs(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.abs = function abs(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.abs();
  };

  AbstractMatrix.prototype.acos = function acos() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.acos(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.acos = function acos(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.acos();
  };

  AbstractMatrix.prototype.acosh = function acosh() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.acosh(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.acosh = function acosh(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.acosh();
  };

  AbstractMatrix.prototype.asin = function asin() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.asin(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.asin = function asin(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.asin();
  };

  AbstractMatrix.prototype.asinh = function asinh() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.asinh(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.asinh = function asinh(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.asinh();
  };

  AbstractMatrix.prototype.atan = function atan() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.atan(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.atan = function atan(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.atan();
  };

  AbstractMatrix.prototype.atanh = function atanh() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.atanh(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.atanh = function atanh(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.atanh();
  };

  AbstractMatrix.prototype.cbrt = function cbrt() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.cbrt(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.cbrt = function cbrt(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.cbrt();
  };

  AbstractMatrix.prototype.ceil = function ceil() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.ceil(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.ceil = function ceil(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.ceil();
  };

  AbstractMatrix.prototype.clz32 = function clz32() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.clz32(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.clz32 = function clz32(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.clz32();
  };

  AbstractMatrix.prototype.cos = function cos() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.cos(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.cos = function cos(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.cos();
  };

  AbstractMatrix.prototype.cosh = function cosh() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.cosh(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.cosh = function cosh(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.cosh();
  };

  AbstractMatrix.prototype.exp = function exp() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.exp(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.exp = function exp(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.exp();
  };

  AbstractMatrix.prototype.expm1 = function expm1() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.expm1(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.expm1 = function expm1(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.expm1();
  };

  AbstractMatrix.prototype.floor = function floor() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.floor(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.floor = function floor(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.floor();
  };

  AbstractMatrix.prototype.fround = function fround() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.fround(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.fround = function fround(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.fround();
  };

  AbstractMatrix.prototype.log = function log() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.log(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.log = function log(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.log();
  };

  AbstractMatrix.prototype.log1p = function log1p() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.log1p(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.log1p = function log1p(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.log1p();
  };

  AbstractMatrix.prototype.log10 = function log10() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.log10(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.log10 = function log10(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.log10();
  };

  AbstractMatrix.prototype.log2 = function log2() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.log2(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.log2 = function log2(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.log2();
  };

  AbstractMatrix.prototype.round = function round() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.round(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.round = function round(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.round();
  };

  AbstractMatrix.prototype.sign = function sign() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.sign(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.sign = function sign(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.sign();
  };

  AbstractMatrix.prototype.sin = function sin() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.sin(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.sin = function sin(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.sin();
  };

  AbstractMatrix.prototype.sinh = function sinh() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.sinh(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.sinh = function sinh(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.sinh();
  };

  AbstractMatrix.prototype.sqrt = function sqrt() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.sqrt(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.sqrt = function sqrt(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.sqrt();
  };

  AbstractMatrix.prototype.tan = function tan() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.tan(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.tan = function tan(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.tan();
  };

  AbstractMatrix.prototype.tanh = function tanh() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.tanh(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.tanh = function tanh(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.tanh();
  };

  AbstractMatrix.prototype.trunc = function trunc() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.trunc(this.get(i, j)));
      }
    }
    return this;
  };

  AbstractMatrix.trunc = function trunc(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.trunc();
  };

  AbstractMatrix.pow = function pow(matrix, arg0) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.pow(arg0);
  };

  AbstractMatrix.prototype.pow = function pow(value) {
    if (typeof value === 'number') return this.powS(value);
    return this.powM(value);
  };

  AbstractMatrix.prototype.powS = function powS(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.pow(this.get(i, j), value));
      }
    }
    return this;
  };

  AbstractMatrix.prototype.powM = function powM(matrix) {
    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, Math.pow(this.get(i, j), matrix.get(i, j)));
      }
    }
    return this;
  };
}

class AbstractMatrix {
  static from1DArray(newRows, newColumns, newData) {
    let length = newRows * newColumns;
    if (length !== newData.length) {
      throw new RangeError('data length does not match given dimensions');
    }
    let newMatrix = new Matrix(newRows, newColumns);
    for (let row = 0; row < newRows; row++) {
      for (let column = 0; column < newColumns; column++) {
        newMatrix.set(row, column, newData[row * newColumns + column]);
      }
    }
    return newMatrix;
  }

  static rowVector(newData) {
    let vector = new Matrix(1, newData.length);
    for (let i = 0; i < newData.length; i++) {
      vector.set(0, i, newData[i]);
    }
    return vector;
  }

  static columnVector(newData) {
    let vector = new Matrix(newData.length, 1);
    for (let i = 0; i < newData.length; i++) {
      vector.set(i, 0, newData[i]);
    }
    return vector;
  }

  static zeros(rows, columns) {
    return new Matrix(rows, columns);
  }

  static ones(rows, columns) {
    return new Matrix(rows, columns).fill(1);
  }

  static rand(rows, columns, options = {}) {
    if (typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const { random = Math.random } = options;
    let matrix = new Matrix(rows, columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        matrix.set(i, j, random());
      }
    }
    return matrix;
  }

  static randInt(rows, columns, options = {}) {
    if (typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const { min = 0, max = 1000, random = Math.random } = options;
    if (!Number.isInteger(min)) throw new TypeError('min must be an integer');
    if (!Number.isInteger(max)) throw new TypeError('max must be an integer');
    if (min >= max) throw new RangeError('min must be smaller than max');
    let interval = max - min;
    let matrix = new Matrix(rows, columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        let value = min + Math.round(random() * interval);
        matrix.set(i, j, value);
      }
    }
    return matrix;
  }

  static eye(rows, columns, value) {
    if (columns === undefined) columns = rows;
    if (value === undefined) value = 1;
    let min = Math.min(rows, columns);
    let matrix = this.zeros(rows, columns);
    for (let i = 0; i < min; i++) {
      matrix.set(i, i, value);
    }
    return matrix;
  }

  static diag(data, rows, columns) {
    let l = data.length;
    if (rows === undefined) rows = l;
    if (columns === undefined) columns = rows;
    let min = Math.min(l, rows, columns);
    let matrix = this.zeros(rows, columns);
    for (let i = 0; i < min; i++) {
      matrix.set(i, i, data[i]);
    }
    return matrix;
  }

  static min(matrix1, matrix2) {
    matrix1 = this.checkMatrix(matrix1);
    matrix2 = this.checkMatrix(matrix2);
    let rows = matrix1.rows;
    let columns = matrix1.columns;
    let result = new Matrix(rows, columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        result.set(i, j, Math.min(matrix1.get(i, j), matrix2.get(i, j)));
      }
    }
    return result;
  }

  static max(matrix1, matrix2) {
    matrix1 = this.checkMatrix(matrix1);
    matrix2 = this.checkMatrix(matrix2);
    let rows = matrix1.rows;
    let columns = matrix1.columns;
    let result = new this(rows, columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        result.set(i, j, Math.max(matrix1.get(i, j), matrix2.get(i, j)));
      }
    }
    return result;
  }

  static checkMatrix(value) {
    return AbstractMatrix.isMatrix(value) ? value : new Matrix(value);
  }

  static isMatrix(value) {
    return value != null && value.klass === 'Matrix';
  }

  get size() {
    return this.rows * this.columns;
  }

  apply(callback) {
    if (typeof callback !== 'function') {
      throw new TypeError('callback must be a function');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        callback.call(this, i, j);
      }
    }
    return this;
  }

  to1DArray() {
    let array = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        array.push(this.get(i, j));
      }
    }
    return array;
  }

  to2DArray() {
    let copy = [];
    for (let i = 0; i < this.rows; i++) {
      copy.push([]);
      for (let j = 0; j < this.columns; j++) {
        copy[i].push(this.get(i, j));
      }
    }
    return copy;
  }

  toJSON() {
    return this.to2DArray();
  }

  isRowVector() {
    return this.rows === 1;
  }

  isColumnVector() {
    return this.columns === 1;
  }

  isVector() {
    return this.rows === 1 || this.columns === 1;
  }

  isSquare() {
    return this.rows === this.columns;
  }

  isSymmetric() {
    if (this.isSquare()) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j <= i; j++) {
          if (this.get(i, j) !== this.get(j, i)) {
            return false;
          }
        }
      }
      return true;
    }
    return false;
  }

  isEchelonForm() {
    let i = 0;
    let j = 0;
    let previousColumn = -1;
    let isEchelonForm = true;
    let checked = false;
    while (i < this.rows && isEchelonForm) {
      j = 0;
      checked = false;
      while (j < this.columns && checked === false) {
        if (this.get(i, j) === 0) {
          j++;
        } else if (this.get(i, j) === 1 && j > previousColumn) {
          checked = true;
          previousColumn = j;
        } else {
          isEchelonForm = false;
          checked = true;
        }
      }
      i++;
    }
    return isEchelonForm;
  }

  isReducedEchelonForm() {
    let i = 0;
    let j = 0;
    let previousColumn = -1;
    let isReducedEchelonForm = true;
    let checked = false;
    while (i < this.rows && isReducedEchelonForm) {
      j = 0;
      checked = false;
      while (j < this.columns && checked === false) {
        if (this.get(i, j) === 0) {
          j++;
        } else if (this.get(i, j) === 1 && j > previousColumn) {
          checked = true;
          previousColumn = j;
        } else {
          isReducedEchelonForm = false;
          checked = true;
        }
      }
      for (let k = j + 1; k < this.rows; k++) {
        if (this.get(i, k) !== 0) {
          isReducedEchelonForm = false;
        }
      }
      i++;
    }
    return isReducedEchelonForm;
  }

  echelonForm() {
    let result = this.clone();
    let h = 0;
    let k = 0;
    while (h < result.rows && k < result.columns) {
      let iMax = h;
      for (let i = h; i < result.rows; i++) {
        if (result.get(i, k) > result.get(iMax, k)) {
          iMax = i;
        }
      }
      if (result.get(iMax, k) === 0) {
        k++;
      } else {
        result.swapRows(h, iMax);
        let tmp = result.get(h, k);
        for (let j = k; j < result.columns; j++) {
          result.set(h, j, result.get(h, j) / tmp);
        }
        for (let i = h + 1; i < result.rows; i++) {
          let factor = result.get(i, k) / result.get(h, k);
          result.set(i, k, 0);
          for (let j = k + 1; j < result.columns; j++) {
            result.set(i, j, result.get(i, j) - result.get(h, j) * factor);
          }
        }
        h++;
        k++;
      }
    }
    return result;
  }

  reducedEchelonForm() {
    let result = this.echelonForm();
    let m = result.columns;
    let n = result.rows;
    let h = n - 1;
    while (h >= 0) {
      if (result.maxRow(h) === 0) {
        h--;
      } else {
        let p = 0;
        let pivot = false;
        while (p < n && pivot === false) {
          if (result.get(h, p) === 1) {
            pivot = true;
          } else {
            p++;
          }
        }
        for (let i = 0; i < h; i++) {
          let factor = result.get(i, p);
          for (let j = p; j < m; j++) {
            let tmp = result.get(i, j) - factor * result.get(h, j);
            result.set(i, j, tmp);
          }
        }
        h--;
      }
    }
    return result;
  }

  set() {
    throw new Error('set method is unimplemented');
  }

  get() {
    throw new Error('get method is unimplemented');
  }

  repeat(options = {}) {
    if (typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const { rows = 1, columns = 1 } = options;
    if (!Number.isInteger(rows) || rows <= 0) {
      throw new TypeError('rows must be a positive integer');
    }
    if (!Number.isInteger(columns) || columns <= 0) {
      throw new TypeError('columns must be a positive integer');
    }
    let matrix = new Matrix(this.rows * rows, this.columns * columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        matrix.setSubMatrix(this, this.rows * i, this.columns * j);
      }
    }
    return matrix;
  }

  fill(value) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, value);
      }
    }
    return this;
  }

  neg() {
    return this.mulS(-1);
  }

  getRow(index) {
    checkRowIndex(this, index);
    let row = [];
    for (let i = 0; i < this.columns; i++) {
      row.push(this.get(index, i));
    }
    return row;
  }

  getRowVector(index) {
    return Matrix.rowVector(this.getRow(index));
  }

  setRow(index, array) {
    checkRowIndex(this, index);
    array = checkRowVector(this, array);
    for (let i = 0; i < this.columns; i++) {
      this.set(index, i, array[i]);
    }
    return this;
  }

  swapRows(row1, row2) {
    checkRowIndex(this, row1);
    checkRowIndex(this, row2);
    for (let i = 0; i < this.columns; i++) {
      let temp = this.get(row1, i);
      this.set(row1, i, this.get(row2, i));
      this.set(row2, i, temp);
    }
    return this;
  }

  getColumn(index) {
    checkColumnIndex(this, index);
    let column = [];
    for (let i = 0; i < this.rows; i++) {
      column.push(this.get(i, index));
    }
    return column;
  }

  getColumnVector(index) {
    return Matrix.columnVector(this.getColumn(index));
  }

  setColumn(index, array) {
    checkColumnIndex(this, index);
    array = checkColumnVector(this, array);
    for (let i = 0; i < this.rows; i++) {
      this.set(i, index, array[i]);
    }
    return this;
  }

  swapColumns(column1, column2) {
    checkColumnIndex(this, column1);
    checkColumnIndex(this, column2);
    for (let i = 0; i < this.rows; i++) {
      let temp = this.get(i, column1);
      this.set(i, column1, this.get(i, column2));
      this.set(i, column2, temp);
    }
    return this;
  }

  addRowVector(vector) {
    vector = checkRowVector(this, vector);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) + vector[j]);
      }
    }
    return this;
  }

  subRowVector(vector) {
    vector = checkRowVector(this, vector);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) - vector[j]);
      }
    }
    return this;
  }

  mulRowVector(vector) {
    vector = checkRowVector(this, vector);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) * vector[j]);
      }
    }
    return this;
  }

  divRowVector(vector) {
    vector = checkRowVector(this, vector);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) / vector[j]);
      }
    }
    return this;
  }

  addColumnVector(vector) {
    vector = checkColumnVector(this, vector);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) + vector[i]);
      }
    }
    return this;
  }

  subColumnVector(vector) {
    vector = checkColumnVector(this, vector);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) - vector[i]);
      }
    }
    return this;
  }

  mulColumnVector(vector) {
    vector = checkColumnVector(this, vector);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) * vector[i]);
      }
    }
    return this;
  }

  divColumnVector(vector) {
    vector = checkColumnVector(this, vector);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, this.get(i, j) / vector[i]);
      }
    }
    return this;
  }

  mulRow(index, value) {
    checkRowIndex(this, index);
    for (let i = 0; i < this.columns; i++) {
      this.set(index, i, this.get(index, i) * value);
    }
    return this;
  }

  mulColumn(index, value) {
    checkColumnIndex(this, index);
    for (let i = 0; i < this.rows; i++) {
      this.set(i, index, this.get(i, index) * value);
    }
    return this;
  }

  max() {
    let v = this.get(0, 0);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        if (this.get(i, j) > v) {
          v = this.get(i, j);
        }
      }
    }
    return v;
  }

  maxIndex() {
    let v = this.get(0, 0);
    let idx = [0, 0];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        if (this.get(i, j) > v) {
          v = this.get(i, j);
          idx[0] = i;
          idx[1] = j;
        }
      }
    }
    return idx;
  }

  min() {
    let v = this.get(0, 0);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        if (this.get(i, j) < v) {
          v = this.get(i, j);
        }
      }
    }
    return v;
  }

  minIndex() {
    let v = this.get(0, 0);
    let idx = [0, 0];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        if (this.get(i, j) < v) {
          v = this.get(i, j);
          idx[0] = i;
          idx[1] = j;
        }
      }
    }
    return idx;
  }

  maxRow(row) {
    checkRowIndex(this, row);
    let v = this.get(row, 0);
    for (let i = 1; i < this.columns; i++) {
      if (this.get(row, i) > v) {
        v = this.get(row, i);
      }
    }
    return v;
  }

  maxRowIndex(row) {
    checkRowIndex(this, row);
    let v = this.get(row, 0);
    let idx = [row, 0];
    for (let i = 1; i < this.columns; i++) {
      if (this.get(row, i) > v) {
        v = this.get(row, i);
        idx[1] = i;
      }
    }
    return idx;
  }

  minRow(row) {
    checkRowIndex(this, row);
    let v = this.get(row, 0);
    for (let i = 1; i < this.columns; i++) {
      if (this.get(row, i) < v) {
        v = this.get(row, i);
      }
    }
    return v;
  }

  minRowIndex(row) {
    checkRowIndex(this, row);
    let v = this.get(row, 0);
    let idx = [row, 0];
    for (let i = 1; i < this.columns; i++) {
      if (this.get(row, i) < v) {
        v = this.get(row, i);
        idx[1] = i;
      }
    }
    return idx;
  }

  maxColumn(column) {
    checkColumnIndex(this, column);
    let v = this.get(0, column);
    for (let i = 1; i < this.rows; i++) {
      if (this.get(i, column) > v) {
        v = this.get(i, column);
      }
    }
    return v;
  }

  maxColumnIndex(column) {
    checkColumnIndex(this, column);
    let v = this.get(0, column);
    let idx = [0, column];
    for (let i = 1; i < this.rows; i++) {
      if (this.get(i, column) > v) {
        v = this.get(i, column);
        idx[0] = i;
      }
    }
    return idx;
  }

  minColumn(column) {
    checkColumnIndex(this, column);
    let v = this.get(0, column);
    for (let i = 1; i < this.rows; i++) {
      if (this.get(i, column) < v) {
        v = this.get(i, column);
      }
    }
    return v;
  }

  minColumnIndex(column) {
    checkColumnIndex(this, column);
    let v = this.get(0, column);
    let idx = [0, column];
    for (let i = 1; i < this.rows; i++) {
      if (this.get(i, column) < v) {
        v = this.get(i, column);
        idx[0] = i;
      }
    }
    return idx;
  }

  diag() {
    let min = Math.min(this.rows, this.columns);
    let diag = [];
    for (let i = 0; i < min; i++) {
      diag.push(this.get(i, i));
    }
    return diag;
  }

  norm(type = 'frobenius') {
    let result = 0;
    if (type === 'max') {
      return this.max();
    } else if (type === 'frobenius') {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          result = result + this.get(i, j) * this.get(i, j);
        }
      }
      return Math.sqrt(result);
    } else {
      throw new RangeError(`unknown norm type: ${type}`);
    }
  }

  cumulativeSum() {
    let sum = 0;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        sum += this.get(i, j);
        this.set(i, j, sum);
      }
    }
    return this;
  }

  dot(vector2) {
    if (AbstractMatrix.isMatrix(vector2)) vector2 = vector2.to1DArray();
    let vector1 = this.to1DArray();
    if (vector1.length !== vector2.length) {
      throw new RangeError('vectors do not have the same size');
    }
    let dot = 0;
    for (let i = 0; i < vector1.length; i++) {
      dot += vector1[i] * vector2[i];
    }
    return dot;
  }

  mmul(other) {
    other = Matrix.checkMatrix(other);

    let m = this.rows;
    let n = this.columns;
    let p = other.columns;

    let result = new Matrix(m, p);

    let Bcolj = new Float64Array(n);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < n; k++) {
        Bcolj[k] = other.get(k, j);
      }

      for (let i = 0; i < m; i++) {
        let s = 0;
        for (let k = 0; k < n; k++) {
          s += this.get(i, k) * Bcolj[k];
        }

        result.set(i, j, s);
      }
    }
    return result;
  }

  strassen2x2(other) {
    other = Matrix.checkMatrix(other);
    let result = new Matrix(2, 2);
    const a11 = this.get(0, 0);
    const b11 = other.get(0, 0);
    const a12 = this.get(0, 1);
    const b12 = other.get(0, 1);
    const a21 = this.get(1, 0);
    const b21 = other.get(1, 0);
    const a22 = this.get(1, 1);
    const b22 = other.get(1, 1);

    // Compute intermediate values.
    const m1 = (a11 + a22) * (b11 + b22);
    const m2 = (a21 + a22) * b11;
    const m3 = a11 * (b12 - b22);
    const m4 = a22 * (b21 - b11);
    const m5 = (a11 + a12) * b22;
    const m6 = (a21 - a11) * (b11 + b12);
    const m7 = (a12 - a22) * (b21 + b22);

    // Combine intermediate values into the output.
    const c00 = m1 + m4 - m5 + m7;
    const c01 = m3 + m5;
    const c10 = m2 + m4;
    const c11 = m1 - m2 + m3 + m6;

    result.set(0, 0, c00);
    result.set(0, 1, c01);
    result.set(1, 0, c10);
    result.set(1, 1, c11);
    return result;
  }

  strassen3x3(other) {
    other = Matrix.checkMatrix(other);
    let result = new Matrix(3, 3);

    const a00 = this.get(0, 0);
    const a01 = this.get(0, 1);
    const a02 = this.get(0, 2);
    const a10 = this.get(1, 0);
    const a11 = this.get(1, 1);
    const a12 = this.get(1, 2);
    const a20 = this.get(2, 0);
    const a21 = this.get(2, 1);
    const a22 = this.get(2, 2);

    const b00 = other.get(0, 0);
    const b01 = other.get(0, 1);
    const b02 = other.get(0, 2);
    const b10 = other.get(1, 0);
    const b11 = other.get(1, 1);
    const b12 = other.get(1, 2);
    const b20 = other.get(2, 0);
    const b21 = other.get(2, 1);
    const b22 = other.get(2, 2);

    const m1 = (a00 + a01 + a02 - a10 - a11 - a21 - a22) * b11;
    const m2 = (a00 - a10) * (-b01 + b11);
    const m3 = a11 * (-b00 + b01 + b10 - b11 - b12 - b20 + b22);
    const m4 = (-a00 + a10 + a11) * (b00 - b01 + b11);
    const m5 = (a10 + a11) * (-b00 + b01);
    const m6 = a00 * b00;
    const m7 = (-a00 + a20 + a21) * (b00 - b02 + b12);
    const m8 = (-a00 + a20) * (b02 - b12);
    const m9 = (a20 + a21) * (-b00 + b02);
    const m10 = (a00 + a01 + a02 - a11 - a12 - a20 - a21) * b12;
    const m11 = a21 * (-b00 + b02 + b10 - b11 - b12 - b20 + b21);
    const m12 = (-a02 + a21 + a22) * (b11 + b20 - b21);
    const m13 = (a02 - a22) * (b11 - b21);
    const m14 = a02 * b20;
    const m15 = (a21 + a22) * (-b20 + b21);
    const m16 = (-a02 + a11 + a12) * (b12 + b20 - b22);
    const m17 = (a02 - a12) * (b12 - b22);
    const m18 = (a11 + a12) * (-b20 + b22);
    const m19 = a01 * b10;
    const m20 = a12 * b21;
    const m21 = a10 * b02;
    const m22 = a20 * b01;
    const m23 = a22 * b22;

    const c00 = m6 + m14 + m19;
    const c01 = m1 + m4 + m5 + m6 + m12 + m14 + m15;
    const c02 = m6 + m7 + m9 + m10 + m14 + m16 + m18;
    const c10 = m2 + m3 + m4 + m6 + m14 + m16 + m17;
    const c11 = m2 + m4 + m5 + m6 + m20;
    const c12 = m14 + m16 + m17 + m18 + m21;
    const c20 = m6 + m7 + m8 + m11 + m12 + m13 + m14;
    const c21 = m12 + m13 + m14 + m15 + m22;
    const c22 = m6 + m7 + m8 + m9 + m23;

    result.set(0, 0, c00);
    result.set(0, 1, c01);
    result.set(0, 2, c02);
    result.set(1, 0, c10);
    result.set(1, 1, c11);
    result.set(1, 2, c12);
    result.set(2, 0, c20);
    result.set(2, 1, c21);
    result.set(2, 2, c22);
    return result;
  }

  mmulStrassen(y) {
    y = Matrix.checkMatrix(y);
    let x = this.clone();
    let r1 = x.rows;
    let c1 = x.columns;
    let r2 = y.rows;
    let c2 = y.columns;
    if (c1 !== r2) {
      // eslint-disable-next-line no-console
      console.warn(
        `Multiplying ${r1} x ${c1} and ${r2} x ${c2} matrix: dimensions do not match.`,
      );
    }

    // Put a matrix into the top left of a matrix of zeros.
    // `rows` and `cols` are the dimensions of the output matrix.
    function embed(mat, rows, cols) {
      let r = mat.rows;
      let c = mat.columns;
      if (r === rows && c === cols) {
        return mat;
      } else {
        let resultat = AbstractMatrix.zeros(rows, cols);
        resultat = resultat.setSubMatrix(mat, 0, 0);
        return resultat;
      }
    }

    // Make sure both matrices are the same size.
    // This is exclusively for simplicity:
    // this algorithm can be implemented with matrices of different sizes.

    let r = Math.max(r1, r2);
    let c = Math.max(c1, c2);
    x = embed(x, r, c);
    y = embed(y, r, c);

    // Our recursive multiplication function.
    function blockMult(a, b, rows, cols) {
      // For small matrices, resort to naive multiplication.
      if (rows <= 512 || cols <= 512) {
        return a.mmul(b); // a is equivalent to this
      }

      // Apply dynamic padding.
      if (rows % 2 === 1 && cols % 2 === 1) {
        a = embed(a, rows + 1, cols + 1);
        b = embed(b, rows + 1, cols + 1);
      } else if (rows % 2 === 1) {
        a = embed(a, rows + 1, cols);
        b = embed(b, rows + 1, cols);
      } else if (cols % 2 === 1) {
        a = embed(a, rows, cols + 1);
        b = embed(b, rows, cols + 1);
      }

      let halfRows = parseInt(a.rows / 2, 10);
      let halfCols = parseInt(a.columns / 2, 10);
      // Subdivide input matrices.
      let a11 = a.subMatrix(0, halfRows - 1, 0, halfCols - 1);
      let b11 = b.subMatrix(0, halfRows - 1, 0, halfCols - 1);

      let a12 = a.subMatrix(0, halfRows - 1, halfCols, a.columns - 1);
      let b12 = b.subMatrix(0, halfRows - 1, halfCols, b.columns - 1);

      let a21 = a.subMatrix(halfRows, a.rows - 1, 0, halfCols - 1);
      let b21 = b.subMatrix(halfRows, b.rows - 1, 0, halfCols - 1);

      let a22 = a.subMatrix(halfRows, a.rows - 1, halfCols, a.columns - 1);
      let b22 = b.subMatrix(halfRows, b.rows - 1, halfCols, b.columns - 1);

      // Compute intermediate values.
      let m1 = blockMult(
        AbstractMatrix.add(a11, a22),
        AbstractMatrix.add(b11, b22),
        halfRows,
        halfCols,
      );
      let m2 = blockMult(AbstractMatrix.add(a21, a22), b11, halfRows, halfCols);
      let m3 = blockMult(a11, AbstractMatrix.sub(b12, b22), halfRows, halfCols);
      let m4 = blockMult(a22, AbstractMatrix.sub(b21, b11), halfRows, halfCols);
      let m5 = blockMult(AbstractMatrix.add(a11, a12), b22, halfRows, halfCols);
      let m6 = blockMult(
        AbstractMatrix.sub(a21, a11),
        AbstractMatrix.add(b11, b12),
        halfRows,
        halfCols,
      );
      let m7 = blockMult(
        AbstractMatrix.sub(a12, a22),
        AbstractMatrix.add(b21, b22),
        halfRows,
        halfCols,
      );

      // Combine intermediate values into the output.
      let c11 = AbstractMatrix.add(m1, m4);
      c11.sub(m5);
      c11.add(m7);
      let c12 = AbstractMatrix.add(m3, m5);
      let c21 = AbstractMatrix.add(m2, m4);
      let c22 = AbstractMatrix.sub(m1, m2);
      c22.add(m3);
      c22.add(m6);

      // Crop output to the desired size (undo dynamic padding).
      let resultat = AbstractMatrix.zeros(2 * c11.rows, 2 * c11.columns);
      resultat = resultat.setSubMatrix(c11, 0, 0);
      resultat = resultat.setSubMatrix(c12, c11.rows, 0);
      resultat = resultat.setSubMatrix(c21, 0, c11.columns);
      resultat = resultat.setSubMatrix(c22, c11.rows, c11.columns);
      return resultat.subMatrix(0, rows - 1, 0, cols - 1);
    }
    return blockMult(x, y, r, c);
  }

  scaleRows(options = {}) {
    if (typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const { min = 0, max = 1 } = options;
    if (!Number.isFinite(min)) throw new TypeError('min must be a number');
    if (!Number.isFinite(max)) throw new TypeError('max must be a number');
    if (min >= max) throw new RangeError('min must be smaller than max');
    let newMatrix = new Matrix(this.rows, this.columns);
    for (let i = 0; i < this.rows; i++) {
      const row = this.getRow(i);
      rescale(row, { min, max, output: row });
      newMatrix.setRow(i, row);
    }
    return newMatrix;
  }

  scaleColumns(options = {}) {
    if (typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const { min = 0, max = 1 } = options;
    if (!Number.isFinite(min)) throw new TypeError('min must be a number');
    if (!Number.isFinite(max)) throw new TypeError('max must be a number');
    if (min >= max) throw new RangeError('min must be smaller than max');
    let newMatrix = new Matrix(this.rows, this.columns);
    for (let i = 0; i < this.columns; i++) {
      const column = this.getColumn(i);
      rescale(column, {
        min: min,
        max: max,
        output: column,
      });
      newMatrix.setColumn(i, column);
    }
    return newMatrix;
  }

  flipRows() {
    const middle = Math.ceil(this.columns / 2);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < middle; j++) {
        let first = this.get(i, j);
        let last = this.get(i, this.columns - 1 - j);
        this.set(i, j, last);
        this.set(i, this.columns - 1 - j, first);
      }
    }
    return this;
  }

  flipColumns() {
    const middle = Math.ceil(this.rows / 2);
    for (let j = 0; j < this.columns; j++) {
      for (let i = 0; i < middle; i++) {
        let first = this.get(i, j);
        let last = this.get(this.rows - 1 - i, j);
        this.set(i, j, last);
        this.set(this.rows - 1 - i, j, first);
      }
    }
    return this;
  }

  kroneckerProduct(other) {
    other = Matrix.checkMatrix(other);

    let m = this.rows;
    let n = this.columns;
    let p = other.rows;
    let q = other.columns;

    let result = new Matrix(m * p, n * q);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < p; k++) {
          for (let l = 0; l < q; l++) {
            result.set(p * i + k, q * j + l, this.get(i, j) * other.get(k, l));
          }
        }
      }
    }
    return result;
  }

  transpose() {
    let result = new Matrix(this.columns, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        result.set(j, i, this.get(i, j));
      }
    }
    return result;
  }

  sortRows(compareFunction = compareNumbers) {
    for (let i = 0; i < this.rows; i++) {
      this.setRow(i, this.getRow(i).sort(compareFunction));
    }
    return this;
  }

  sortColumns(compareFunction = compareNumbers) {
    for (let i = 0; i < this.columns; i++) {
      this.setColumn(i, this.getColumn(i).sort(compareFunction));
    }
    return this;
  }

  subMatrix(startRow, endRow, startColumn, endColumn) {
    checkRange(this, startRow, endRow, startColumn, endColumn);
    let newMatrix = new Matrix(
      endRow - startRow + 1,
      endColumn - startColumn + 1,
    );
    for (let i = startRow; i <= endRow; i++) {
      for (let j = startColumn; j <= endColumn; j++) {
        newMatrix.set(i - startRow, j - startColumn, this.get(i, j));
      }
    }
    return newMatrix;
  }

  subMatrixRow(indices, startColumn, endColumn) {
    if (startColumn === undefined) startColumn = 0;
    if (endColumn === undefined) endColumn = this.columns - 1;
    if (
      startColumn > endColumn ||
      startColumn < 0 ||
      startColumn >= this.columns ||
      endColumn < 0 ||
      endColumn >= this.columns
    ) {
      throw new RangeError('Argument out of range');
    }

    let newMatrix = new Matrix(indices.length, endColumn - startColumn + 1);
    for (let i = 0; i < indices.length; i++) {
      for (let j = startColumn; j <= endColumn; j++) {
        if (indices[i] < 0 || indices[i] >= this.rows) {
          throw new RangeError(`Row index out of range: ${indices[i]}`);
        }
        newMatrix.set(i, j - startColumn, this.get(indices[i], j));
      }
    }
    return newMatrix;
  }

  subMatrixColumn(indices, startRow, endRow) {
    if (startRow === undefined) startRow = 0;
    if (endRow === undefined) endRow = this.rows - 1;
    if (
      startRow > endRow ||
      startRow < 0 ||
      startRow >= this.rows ||
      endRow < 0 ||
      endRow >= this.rows
    ) {
      throw new RangeError('Argument out of range');
    }

    let newMatrix = new Matrix(endRow - startRow + 1, indices.length);
    for (let i = 0; i < indices.length; i++) {
      for (let j = startRow; j <= endRow; j++) {
        if (indices[i] < 0 || indices[i] >= this.columns) {
          throw new RangeError(`Column index out of range: ${indices[i]}`);
        }
        newMatrix.set(j - startRow, i, this.get(j, indices[i]));
      }
    }
    return newMatrix;
  }

  setSubMatrix(matrix, startRow, startColumn) {
    matrix = Matrix.checkMatrix(matrix);
    let endRow = startRow + matrix.rows - 1;
    let endColumn = startColumn + matrix.columns - 1;
    checkRange(this, startRow, endRow, startColumn, endColumn);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        this.set(startRow + i, startColumn + j, matrix.get(i, j));
      }
    }
    return this;
  }

  selection(rowIndices, columnIndices) {
    let indices = checkIndices(this, rowIndices, columnIndices);
    let newMatrix = new Matrix(rowIndices.length, columnIndices.length);
    for (let i = 0; i < indices.row.length; i++) {
      let rowIndex = indices.row[i];
      for (let j = 0; j < indices.column.length; j++) {
        let columnIndex = indices.column[j];
        newMatrix.set(i, j, this.get(rowIndex, columnIndex));
      }
    }
    return newMatrix;
  }

  trace() {
    let min = Math.min(this.rows, this.columns);
    let trace = 0;
    for (let i = 0; i < min; i++) {
      trace += this.get(i, i);
    }
    return trace;
  }

  clone() {
    let newMatrix = new Matrix(this.rows, this.columns);
    for (let row = 0; row < this.rows; row++) {
      for (let column = 0; column < this.columns; column++) {
        newMatrix.set(row, column, this.get(row, column));
      }
    }
    return newMatrix;
  }

  sum(by) {
    switch (by) {
      case 'row':
        return sumByRow(this);
      case 'column':
        return sumByColumn(this);
      case undefined:
        return sumAll(this);
      default:
        throw new Error(`invalid option: ${by}`);
    }
  }

  product(by) {
    switch (by) {
      case 'row':
        return productByRow(this);
      case 'column':
        return productByColumn(this);
      case undefined:
        return productAll(this);
      default:
        throw new Error(`invalid option: ${by}`);
    }
  }

  mean(by) {
    const sum = this.sum(by);
    switch (by) {
      case 'row': {
        for (let i = 0; i < this.rows; i++) {
          sum[i] /= this.columns;
        }
        return sum;
      }
      case 'column': {
        for (let i = 0; i < this.columns; i++) {
          sum[i] /= this.rows;
        }
        return sum;
      }
      case undefined:
        return sum / this.size;
      default:
        throw new Error(`invalid option: ${by}`);
    }
  }

  variance(by, options = {}) {
    if (typeof by === 'object') {
      options = by;
      by = undefined;
    }
    if (typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const { unbiased = true, mean = this.mean(by) } = options;
    if (typeof unbiased !== 'boolean') {
      throw new TypeError('unbiased must be a boolean');
    }
    switch (by) {
      case 'row': {
        if (!Array.isArray(mean)) {
          throw new TypeError('mean must be an array');
        }
        return varianceByRow(this, unbiased, mean);
      }
      case 'column': {
        if (!Array.isArray(mean)) {
          throw new TypeError('mean must be an array');
        }
        return varianceByColumn(this, unbiased, mean);
      }
      case undefined: {
        if (typeof mean !== 'number') {
          throw new TypeError('mean must be a number');
        }
        return varianceAll(this, unbiased, mean);
      }
      default:
        throw new Error(`invalid option: ${by}`);
    }
  }

  standardDeviation(by, options) {
    if (typeof by === 'object') {
      options = by;
      by = undefined;
    }
    const variance = this.variance(by, options);
    if (by === undefined) {
      return Math.sqrt(variance);
    } else {
      for (let i = 0; i < variance.length; i++) {
        variance[i] = Math.sqrt(variance[i]);
      }
      return variance;
    }
  }

  center(by, options = {}) {
    if (typeof by === 'object') {
      options = by;
      by = undefined;
    }
    if (typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const { center = this.mean(by) } = options;
    switch (by) {
      case 'row': {
        if (!Array.isArray(center)) {
          throw new TypeError('center must be an array');
        }
        centerByRow(this, center);
        return this;
      }
      case 'column': {
        if (!Array.isArray(center)) {
          throw new TypeError('center must be an array');
        }
        centerByColumn(this, center);
        return this;
      }
      case undefined: {
        if (typeof center !== 'number') {
          throw new TypeError('center must be a number');
        }
        centerAll(this, center);
        return this;
      }
      default:
        throw new Error(`invalid option: ${by}`);
    }
  }

  scale(by, options = {}) {
    if (typeof by === 'object') {
      options = by;
      by = undefined;
    }
    if (typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    let scale = options.scale;
    switch (by) {
      case 'row': {
        if (scale === undefined) {
          scale = getScaleByRow(this);
        } else if (!Array.isArray(scale)) {
          throw new TypeError('scale must be an array');
        }
        scaleByRow(this, scale);
        return this;
      }
      case 'column': {
        if (scale === undefined) {
          scale = getScaleByColumn(this);
        } else if (!Array.isArray(scale)) {
          throw new TypeError('scale must be an array');
        }
        scaleByColumn(this, scale);
        return this;
      }
      case undefined: {
        if (scale === undefined) {
          scale = getScaleAll(this);
        } else if (typeof scale !== 'number') {
          throw new TypeError('scale must be a number');
        }
        scaleAll(this, scale);
        return this;
      }
      default:
        throw new Error(`invalid option: ${by}`);
    }
  }
}

AbstractMatrix.prototype.klass = 'Matrix';
if (typeof Symbol !== 'undefined') {
  AbstractMatrix.prototype[
    Symbol.for('nodejs.util.inspect.custom')
  ] = inspectMatrix;
}

function compareNumbers(a, b) {
  return a - b;
}

// Synonyms
AbstractMatrix.random = AbstractMatrix.rand;
AbstractMatrix.randomInt = AbstractMatrix.randInt;
AbstractMatrix.diagonal = AbstractMatrix.diag;
AbstractMatrix.prototype.diagonal = AbstractMatrix.prototype.diag;
AbstractMatrix.identity = AbstractMatrix.eye;
AbstractMatrix.prototype.negate = AbstractMatrix.prototype.neg;
AbstractMatrix.prototype.tensorProduct =
  AbstractMatrix.prototype.kroneckerProduct;

class Matrix extends AbstractMatrix {
  constructor(nRows, nColumns) {
    super();
    if (Matrix.isMatrix(nRows)) {
      return nRows.clone();
    } else if (Number.isInteger(nRows) && nRows > 0) {
      // Create an empty matrix
      this.data = [];
      if (Number.isInteger(nColumns) && nColumns > 0) {
        for (let i = 0; i < nRows; i++) {
          this.data.push(new Float64Array(nColumns));
        }
      } else {
        throw new TypeError('nColumns must be a positive integer');
      }
    } else if (Array.isArray(nRows)) {
      // Copy the values from the 2D array
      const arrayData = nRows;
      nRows = arrayData.length;
      nColumns = arrayData[0].length;
      if (typeof nColumns !== 'number' || nColumns === 0) {
        throw new TypeError(
          'Data must be a 2D array with at least one element',
        );
      }
      this.data = [];
      for (let i = 0; i < nRows; i++) {
        if (arrayData[i].length !== nColumns) {
          throw new RangeError('Inconsistent array dimensions');
        }
        this.data.push(Float64Array.from(arrayData[i]));
      }
    } else {
      throw new TypeError(
        'First argument must be a positive number or an array',
      );
    }
    this.rows = nRows;
    this.columns = nColumns;
    return this;
  }

  set(rowIndex, columnIndex, value) {
    this.data[rowIndex][columnIndex] = value;
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.data[rowIndex][columnIndex];
  }

  removeRow(index) {
    checkRowIndex(this, index);
    if (this.rows === 1) {
      throw new RangeError('A matrix cannot have less than one row');
    }
    this.data.splice(index, 1);
    this.rows -= 1;
    return this;
  }

  addRow(index, array) {
    if (array === undefined) {
      array = index;
      index = this.rows;
    }
    checkRowIndex(this, index, true);
    array = Float64Array.from(checkRowVector(this, array));
    this.data.splice(index, 0, array);
    this.rows += 1;
    return this;
  }

  removeColumn(index) {
    checkColumnIndex(this, index);
    if (this.columns === 1) {
      throw new RangeError('A matrix cannot have less than one column');
    }
    for (let i = 0; i < this.rows; i++) {
      const newRow = new Float64Array(this.columns - 1);
      for (let j = 0; j < index; j++) {
        newRow[j] = this.data[i][j];
      }
      for (let j = index + 1; j < this.columns; j++) {
        newRow[j - 1] = this.data[i][j];
      }
      this.data[i] = newRow;
    }
    this.columns -= 1;
    return this;
  }

  addColumn(index, array) {
    if (typeof array === 'undefined') {
      array = index;
      index = this.columns;
    }
    checkColumnIndex(this, index, true);
    array = checkColumnVector(this, array);
    for (let i = 0; i < this.rows; i++) {
      const newRow = new Float64Array(this.columns + 1);
      let j = 0;
      for (; j < index; j++) {
        newRow[j] = this.data[i][j];
      }
      newRow[j++] = array[i];
      for (; j < this.columns + 1; j++) {
        newRow[j] = this.data[i][j - 1];
      }
      this.data[i] = newRow;
    }
    this.columns += 1;
    return this;
  }
}

installMathOperations(AbstractMatrix, Matrix);

class BaseView extends AbstractMatrix {
  constructor(matrix, rows, columns) {
    super();
    this.matrix = matrix;
    this.rows = rows;
    this.columns = columns;
  }
}

class MatrixColumnView extends BaseView {
  constructor(matrix, column) {
    checkColumnIndex(matrix, column);
    super(matrix, matrix.rows, 1);
    this.column = column;
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(rowIndex, this.column, value);
    return this;
  }

  get(rowIndex) {
    return this.matrix.get(rowIndex, this.column);
  }
}

class MatrixColumnSelectionView extends BaseView {
  constructor(matrix, columnIndices) {
    columnIndices = checkColumnIndices(matrix, columnIndices);
    super(matrix, matrix.rows, columnIndices.length);
    this.columnIndices = columnIndices;
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(rowIndex, this.columnIndices[columnIndex], value);
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.matrix.get(rowIndex, this.columnIndices[columnIndex]);
  }
}

class MatrixFlipColumnView extends BaseView {
  constructor(matrix) {
    super(matrix, matrix.rows, matrix.columns);
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(rowIndex, this.columns - columnIndex - 1, value);
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.matrix.get(rowIndex, this.columns - columnIndex - 1);
  }
}

class MatrixFlipRowView extends BaseView {
  constructor(matrix) {
    super(matrix, matrix.rows, matrix.columns);
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(this.rows - rowIndex - 1, columnIndex, value);
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.matrix.get(this.rows - rowIndex - 1, columnIndex);
  }
}

class MatrixRowView extends BaseView {
  constructor(matrix, row) {
    checkRowIndex(matrix, row);
    super(matrix, 1, matrix.columns);
    this.row = row;
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(this.row, columnIndex, value);
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.matrix.get(this.row, columnIndex);
  }
}

class MatrixRowSelectionView extends BaseView {
  constructor(matrix, rowIndices) {
    rowIndices = checkRowIndices(matrix, rowIndices);
    super(matrix, rowIndices.length, matrix.columns);
    this.rowIndices = rowIndices;
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(this.rowIndices[rowIndex], columnIndex, value);
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.matrix.get(this.rowIndices[rowIndex], columnIndex);
  }
}

class MatrixSelectionView extends BaseView {
  constructor(matrix, rowIndices, columnIndices) {
    let indices = checkIndices(matrix, rowIndices, columnIndices);
    super(matrix, indices.row.length, indices.column.length);
    this.rowIndices = indices.row;
    this.columnIndices = indices.column;
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(
      this.rowIndices[rowIndex],
      this.columnIndices[columnIndex],
      value,
    );
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.matrix.get(
      this.rowIndices[rowIndex],
      this.columnIndices[columnIndex],
    );
  }
}

class MatrixSubView extends BaseView {
  constructor(matrix, startRow, endRow, startColumn, endColumn) {
    checkRange(matrix, startRow, endRow, startColumn, endColumn);
    super(matrix, endRow - startRow + 1, endColumn - startColumn + 1);
    this.startRow = startRow;
    this.startColumn = startColumn;
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(
      this.startRow + rowIndex,
      this.startColumn + columnIndex,
      value,
    );
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.matrix.get(
      this.startRow + rowIndex,
      this.startColumn + columnIndex,
    );
  }
}

class MatrixTransposeView extends BaseView {
  constructor(matrix) {
    super(matrix, matrix.columns, matrix.rows);
  }

  set(rowIndex, columnIndex, value) {
    this.matrix.set(columnIndex, rowIndex, value);
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.matrix.get(columnIndex, rowIndex);
  }
}

class WrapperMatrix1D extends AbstractMatrix {
  constructor(data, options = {}) {
    const { rows = 1 } = options;

    if (data.length % rows !== 0) {
      throw new Error('the data length is not divisible by the number of rows');
    }
    super();
    this.rows = rows;
    this.columns = data.length / rows;
    this.data = data;
  }

  set(rowIndex, columnIndex, value) {
    let index = this._calculateIndex(rowIndex, columnIndex);
    this.data[index] = value;
    return this;
  }

  get(rowIndex, columnIndex) {
    let index = this._calculateIndex(rowIndex, columnIndex);
    return this.data[index];
  }

  _calculateIndex(row, column) {
    return row * this.columns + column;
  }
}

class WrapperMatrix2D extends AbstractMatrix {
  constructor(data) {
    super();
    this.data = data;
    this.rows = data.length;
    this.columns = data[0].length;
  }

  set(rowIndex, columnIndex, value) {
    this.data[rowIndex][columnIndex] = value;
    return this;
  }

  get(rowIndex, columnIndex) {
    return this.data[rowIndex][columnIndex];
  }
}

function wrap(array, options) {
  if (Array.isArray(array)) {
    if (array[0] && Array.isArray(array[0])) {
      return new WrapperMatrix2D(array);
    } else {
      return new WrapperMatrix1D(array, options);
    }
  } else {
    throw new Error('the argument is not an array');
  }
}

class LuDecomposition {
  constructor(matrix) {
    matrix = WrapperMatrix2D.checkMatrix(matrix);

    let lu = matrix.clone();
    let rows = lu.rows;
    let columns = lu.columns;
    let pivotVector = new Float64Array(rows);
    let pivotSign = 1;
    let i, j, k, p, s, t, v;
    let LUcolj, kmax;

    for (i = 0; i < rows; i++) {
      pivotVector[i] = i;
    }

    LUcolj = new Float64Array(rows);

    for (j = 0; j < columns; j++) {
      for (i = 0; i < rows; i++) {
        LUcolj[i] = lu.get(i, j);
      }

      for (i = 0; i < rows; i++) {
        kmax = Math.min(i, j);
        s = 0;
        for (k = 0; k < kmax; k++) {
          s += lu.get(i, k) * LUcolj[k];
        }
        LUcolj[i] -= s;
        lu.set(i, j, LUcolj[i]);
      }

      p = j;
      for (i = j + 1; i < rows; i++) {
        if (Math.abs(LUcolj[i]) > Math.abs(LUcolj[p])) {
          p = i;
        }
      }

      if (p !== j) {
        for (k = 0; k < columns; k++) {
          t = lu.get(p, k);
          lu.set(p, k, lu.get(j, k));
          lu.set(j, k, t);
        }

        v = pivotVector[p];
        pivotVector[p] = pivotVector[j];
        pivotVector[j] = v;

        pivotSign = -pivotSign;
      }

      if (j < rows && lu.get(j, j) !== 0) {
        for (i = j + 1; i < rows; i++) {
          lu.set(i, j, lu.get(i, j) / lu.get(j, j));
        }
      }
    }

    this.LU = lu;
    this.pivotVector = pivotVector;
    this.pivotSign = pivotSign;
  }

  isSingular() {
    let data = this.LU;
    let col = data.columns;
    for (let j = 0; j < col; j++) {
      if (data.get(j, j) === 0) {
        return true;
      }
    }
    return false;
  }

  solve(value) {
    value = Matrix.checkMatrix(value);

    let lu = this.LU;
    let rows = lu.rows;

    if (rows !== value.rows) {
      throw new Error('Invalid matrix dimensions');
    }
    if (this.isSingular()) {
      throw new Error('LU matrix is singular');
    }

    let count = value.columns;
    let X = value.subMatrixRow(this.pivotVector, 0, count - 1);
    let columns = lu.columns;
    let i, j, k;

    for (k = 0; k < columns; k++) {
      for (i = k + 1; i < columns; i++) {
        for (j = 0; j < count; j++) {
          X.set(i, j, X.get(i, j) - X.get(k, j) * lu.get(i, k));
        }
      }
    }
    for (k = columns - 1; k >= 0; k--) {
      for (j = 0; j < count; j++) {
        X.set(k, j, X.get(k, j) / lu.get(k, k));
      }
      for (i = 0; i < k; i++) {
        for (j = 0; j < count; j++) {
          X.set(i, j, X.get(i, j) - X.get(k, j) * lu.get(i, k));
        }
      }
    }
    return X;
  }

  get determinant() {
    let data = this.LU;
    if (!data.isSquare()) {
      throw new Error('Matrix must be square');
    }
    let determinant = this.pivotSign;
    let col = data.columns;
    for (let j = 0; j < col; j++) {
      determinant *= data.get(j, j);
    }
    return determinant;
  }

  get lowerTriangularMatrix() {
    let data = this.LU;
    let rows = data.rows;
    let columns = data.columns;
    let X = new Matrix(rows, columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        if (i > j) {
          X.set(i, j, data.get(i, j));
        } else if (i === j) {
          X.set(i, j, 1);
        } else {
          X.set(i, j, 0);
        }
      }
    }
    return X;
  }

  get upperTriangularMatrix() {
    let data = this.LU;
    let rows = data.rows;
    let columns = data.columns;
    let X = new Matrix(rows, columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        if (i <= j) {
          X.set(i, j, data.get(i, j));
        } else {
          X.set(i, j, 0);
        }
      }
    }
    return X;
  }

  get pivotPermutationVector() {
    return Array.from(this.pivotVector);
  }
}

function hypotenuse(a, b) {
  let r = 0;
  if (Math.abs(a) > Math.abs(b)) {
    r = b / a;
    return Math.abs(a) * Math.sqrt(1 + r * r);
  }
  if (b !== 0) {
    r = a / b;
    return Math.abs(b) * Math.sqrt(1 + r * r);
  }
  return 0;
}

class QrDecomposition {
  constructor(value) {
    value = WrapperMatrix2D.checkMatrix(value);

    let qr = value.clone();
    let m = value.rows;
    let n = value.columns;
    let rdiag = new Float64Array(n);
    let i, j, k, s;

    for (k = 0; k < n; k++) {
      let nrm = 0;
      for (i = k; i < m; i++) {
        nrm = hypotenuse(nrm, qr.get(i, k));
      }
      if (nrm !== 0) {
        if (qr.get(k, k) < 0) {
          nrm = -nrm;
        }
        for (i = k; i < m; i++) {
          qr.set(i, k, qr.get(i, k) / nrm);
        }
        qr.set(k, k, qr.get(k, k) + 1);
        for (j = k + 1; j < n; j++) {
          s = 0;
          for (i = k; i < m; i++) {
            s += qr.get(i, k) * qr.get(i, j);
          }
          s = -s / qr.get(k, k);
          for (i = k; i < m; i++) {
            qr.set(i, j, qr.get(i, j) + s * qr.get(i, k));
          }
        }
      }
      rdiag[k] = -nrm;
    }

    this.QR = qr;
    this.Rdiag = rdiag;
  }

  solve(value) {
    value = Matrix.checkMatrix(value);

    let qr = this.QR;
    let m = qr.rows;

    if (value.rows !== m) {
      throw new Error('Matrix row dimensions must agree');
    }
    if (!this.isFullRank()) {
      throw new Error('Matrix is rank deficient');
    }

    let count = value.columns;
    let X = value.clone();
    let n = qr.columns;
    let i, j, k, s;

    for (k = 0; k < n; k++) {
      for (j = 0; j < count; j++) {
        s = 0;
        for (i = k; i < m; i++) {
          s += qr.get(i, k) * X.get(i, j);
        }
        s = -s / qr.get(k, k);
        for (i = k; i < m; i++) {
          X.set(i, j, X.get(i, j) + s * qr.get(i, k));
        }
      }
    }
    for (k = n - 1; k >= 0; k--) {
      for (j = 0; j < count; j++) {
        X.set(k, j, X.get(k, j) / this.Rdiag[k]);
      }
      for (i = 0; i < k; i++) {
        for (j = 0; j < count; j++) {
          X.set(i, j, X.get(i, j) - X.get(k, j) * qr.get(i, k));
        }
      }
    }

    return X.subMatrix(0, n - 1, 0, count - 1);
  }

  isFullRank() {
    let columns = this.QR.columns;
    for (let i = 0; i < columns; i++) {
      if (this.Rdiag[i] === 0) {
        return false;
      }
    }
    return true;
  }

  get upperTriangularMatrix() {
    let qr = this.QR;
    let n = qr.columns;
    let X = new Matrix(n, n);
    let i, j;
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        if (i < j) {
          X.set(i, j, qr.get(i, j));
        } else if (i === j) {
          X.set(i, j, this.Rdiag[i]);
        } else {
          X.set(i, j, 0);
        }
      }
    }
    return X;
  }

  get orthogonalMatrix() {
    let qr = this.QR;
    let rows = qr.rows;
    let columns = qr.columns;
    let X = new Matrix(rows, columns);
    let i, j, k, s;

    for (k = columns - 1; k >= 0; k--) {
      for (i = 0; i < rows; i++) {
        X.set(i, k, 0);
      }
      X.set(k, k, 1);
      for (j = k; j < columns; j++) {
        if (qr.get(k, k) !== 0) {
          s = 0;
          for (i = k; i < rows; i++) {
            s += qr.get(i, k) * X.get(i, j);
          }

          s = -s / qr.get(k, k);

          for (i = k; i < rows; i++) {
            X.set(i, j, X.get(i, j) + s * qr.get(i, k));
          }
        }
      }
    }
    return X;
  }
}

class SingularValueDecomposition {
  constructor(value, options = {}) {
    value = WrapperMatrix2D.checkMatrix(value);

    let m = value.rows;
    let n = value.columns;

    const {
      computeLeftSingularVectors = true,
      computeRightSingularVectors = true,
      autoTranspose = false,
    } = options;

    let wantu = Boolean(computeLeftSingularVectors);
    let wantv = Boolean(computeRightSingularVectors);

    let swapped = false;
    let a;
    if (m < n) {
      if (!autoTranspose) {
        a = value.clone();
        // eslint-disable-next-line no-console
        console.warn(
          'Computing SVD on a matrix with more columns than rows. Consider enabling autoTranspose',
        );
      } else {
        a = value.transpose();
        m = a.rows;
        n = a.columns;
        swapped = true;
        let aux = wantu;
        wantu = wantv;
        wantv = aux;
      }
    } else {
      a = value.clone();
    }

    let nu = Math.min(m, n);
    let ni = Math.min(m + 1, n);
    let s = new Float64Array(ni);
    let U = new Matrix(m, nu);
    let V = new Matrix(n, n);

    let e = new Float64Array(n);
    let work = new Float64Array(m);

    let si = new Float64Array(ni);
    for (let i = 0; i < ni; i++) si[i] = i;

    let nct = Math.min(m - 1, n);
    let nrt = Math.max(0, Math.min(n - 2, m));
    let mrc = Math.max(nct, nrt);

    for (let k = 0; k < mrc; k++) {
      if (k < nct) {
        s[k] = 0;
        for (let i = k; i < m; i++) {
          s[k] = hypotenuse(s[k], a.get(i, k));
        }
        if (s[k] !== 0) {
          if (a.get(k, k) < 0) {
            s[k] = -s[k];
          }
          for (let i = k; i < m; i++) {
            a.set(i, k, a.get(i, k) / s[k]);
          }
          a.set(k, k, a.get(k, k) + 1);
        }
        s[k] = -s[k];
      }

      for (let j = k + 1; j < n; j++) {
        if (k < nct && s[k] !== 0) {
          let t = 0;
          for (let i = k; i < m; i++) {
            t += a.get(i, k) * a.get(i, j);
          }
          t = -t / a.get(k, k);
          for (let i = k; i < m; i++) {
            a.set(i, j, a.get(i, j) + t * a.get(i, k));
          }
        }
        e[j] = a.get(k, j);
      }

      if (wantu && k < nct) {
        for (let i = k; i < m; i++) {
          U.set(i, k, a.get(i, k));
        }
      }

      if (k < nrt) {
        e[k] = 0;
        for (let i = k + 1; i < n; i++) {
          e[k] = hypotenuse(e[k], e[i]);
        }
        if (e[k] !== 0) {
          if (e[k + 1] < 0) {
            e[k] = 0 - e[k];
          }
          for (let i = k + 1; i < n; i++) {
            e[i] /= e[k];
          }
          e[k + 1] += 1;
        }
        e[k] = -e[k];
        if (k + 1 < m && e[k] !== 0) {
          for (let i = k + 1; i < m; i++) {
            work[i] = 0;
          }
          for (let i = k + 1; i < m; i++) {
            for (let j = k + 1; j < n; j++) {
              work[i] += e[j] * a.get(i, j);
            }
          }
          for (let j = k + 1; j < n; j++) {
            let t = -e[j] / e[k + 1];
            for (let i = k + 1; i < m; i++) {
              a.set(i, j, a.get(i, j) + t * work[i]);
            }
          }
        }
        if (wantv) {
          for (let i = k + 1; i < n; i++) {
            V.set(i, k, e[i]);
          }
        }
      }
    }

    let p = Math.min(n, m + 1);
    if (nct < n) {
      s[nct] = a.get(nct, nct);
    }
    if (m < p) {
      s[p - 1] = 0;
    }
    if (nrt + 1 < p) {
      e[nrt] = a.get(nrt, p - 1);
    }
    e[p - 1] = 0;

    if (wantu) {
      for (let j = nct; j < nu; j++) {
        for (let i = 0; i < m; i++) {
          U.set(i, j, 0);
        }
        U.set(j, j, 1);
      }
      for (let k = nct - 1; k >= 0; k--) {
        if (s[k] !== 0) {
          for (let j = k + 1; j < nu; j++) {
            let t = 0;
            for (let i = k; i < m; i++) {
              t += U.get(i, k) * U.get(i, j);
            }
            t = -t / U.get(k, k);
            for (let i = k; i < m; i++) {
              U.set(i, j, U.get(i, j) + t * U.get(i, k));
            }
          }
          for (let i = k; i < m; i++) {
            U.set(i, k, -U.get(i, k));
          }
          U.set(k, k, 1 + U.get(k, k));
          for (let i = 0; i < k - 1; i++) {
            U.set(i, k, 0);
          }
        } else {
          for (let i = 0; i < m; i++) {
            U.set(i, k, 0);
          }
          U.set(k, k, 1);
        }
      }
    }

    if (wantv) {
      for (let k = n - 1; k >= 0; k--) {
        if (k < nrt && e[k] !== 0) {
          for (let j = k + 1; j < n; j++) {
            let t = 0;
            for (let i = k + 1; i < n; i++) {
              t += V.get(i, k) * V.get(i, j);
            }
            t = -t / V.get(k + 1, k);
            for (let i = k + 1; i < n; i++) {
              V.set(i, j, V.get(i, j) + t * V.get(i, k));
            }
          }
        }
        for (let i = 0; i < n; i++) {
          V.set(i, k, 0);
        }
        V.set(k, k, 1);
      }
    }

    let pp = p - 1;
    let eps = Number.EPSILON;
    while (p > 0) {
      let k, kase;
      for (k = p - 2; k >= -1; k--) {
        if (k === -1) {
          break;
        }
        const alpha =
          Number.MIN_VALUE + eps * Math.abs(s[k] + Math.abs(s[k + 1]));
        if (Math.abs(e[k]) <= alpha || Number.isNaN(e[k])) {
          e[k] = 0;
          break;
        }
      }
      if (k === p - 2) {
        kase = 4;
      } else {
        let ks;
        for (ks = p - 1; ks >= k; ks--) {
          if (ks === k) {
            break;
          }
          let t =
            (ks !== p ? Math.abs(e[ks]) : 0) +
            (ks !== k + 1 ? Math.abs(e[ks - 1]) : 0);
          if (Math.abs(s[ks]) <= eps * t) {
            s[ks] = 0;
            break;
          }
        }
        if (ks === k) {
          kase = 3;
        } else if (ks === p - 1) {
          kase = 1;
        } else {
          kase = 2;
          k = ks;
        }
      }

      k++;

      switch (kase) {
        case 1: {
          let f = e[p - 2];
          e[p - 2] = 0;
          for (let j = p - 2; j >= k; j--) {
            let t = hypotenuse(s[j], f);
            let cs = s[j] / t;
            let sn = f / t;
            s[j] = t;
            if (j !== k) {
              f = -sn * e[j - 1];
              e[j - 1] = cs * e[j - 1];
            }
            if (wantv) {
              for (let i = 0; i < n; i++) {
                t = cs * V.get(i, j) + sn * V.get(i, p - 1);
                V.set(i, p - 1, -sn * V.get(i, j) + cs * V.get(i, p - 1));
                V.set(i, j, t);
              }
            }
          }
          break;
        }
        case 2: {
          let f = e[k - 1];
          e[k - 1] = 0;
          for (let j = k; j < p; j++) {
            let t = hypotenuse(s[j], f);
            let cs = s[j] / t;
            let sn = f / t;
            s[j] = t;
            f = -sn * e[j];
            e[j] = cs * e[j];
            if (wantu) {
              for (let i = 0; i < m; i++) {
                t = cs * U.get(i, j) + sn * U.get(i, k - 1);
                U.set(i, k - 1, -sn * U.get(i, j) + cs * U.get(i, k - 1));
                U.set(i, j, t);
              }
            }
          }
          break;
        }
        case 3: {
          const scale = Math.max(
            Math.abs(s[p - 1]),
            Math.abs(s[p - 2]),
            Math.abs(e[p - 2]),
            Math.abs(s[k]),
            Math.abs(e[k]),
          );
          const sp = s[p - 1] / scale;
          const spm1 = s[p - 2] / scale;
          const epm1 = e[p - 2] / scale;
          const sk = s[k] / scale;
          const ek = e[k] / scale;
          const b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2;
          const c = sp * epm1 * (sp * epm1);
          let shift = 0;
          if (b !== 0 || c !== 0) {
            if (b < 0) {
              shift = 0 - Math.sqrt(b * b + c);
            } else {
              shift = Math.sqrt(b * b + c);
            }
            shift = c / (b + shift);
          }
          let f = (sk + sp) * (sk - sp) + shift;
          let g = sk * ek;
          for (let j = k; j < p - 1; j++) {
            let t = hypotenuse(f, g);
            if (t === 0) t = Number.MIN_VALUE;
            let cs = f / t;
            let sn = g / t;
            if (j !== k) {
              e[j - 1] = t;
            }
            f = cs * s[j] + sn * e[j];
            e[j] = cs * e[j] - sn * s[j];
            g = sn * s[j + 1];
            s[j + 1] = cs * s[j + 1];
            if (wantv) {
              for (let i = 0; i < n; i++) {
                t = cs * V.get(i, j) + sn * V.get(i, j + 1);
                V.set(i, j + 1, -sn * V.get(i, j) + cs * V.get(i, j + 1));
                V.set(i, j, t);
              }
            }
            t = hypotenuse(f, g);
            if (t === 0) t = Number.MIN_VALUE;
            cs = f / t;
            sn = g / t;
            s[j] = t;
            f = cs * e[j] + sn * s[j + 1];
            s[j + 1] = -sn * e[j] + cs * s[j + 1];
            g = sn * e[j + 1];
            e[j + 1] = cs * e[j + 1];
            if (wantu && j < m - 1) {
              for (let i = 0; i < m; i++) {
                t = cs * U.get(i, j) + sn * U.get(i, j + 1);
                U.set(i, j + 1, -sn * U.get(i, j) + cs * U.get(i, j + 1));
                U.set(i, j, t);
              }
            }
          }
          e[p - 2] = f;
          break;
        }
        case 4: {
          if (s[k] <= 0) {
            s[k] = s[k] < 0 ? -s[k] : 0;
            if (wantv) {
              for (let i = 0; i <= pp; i++) {
                V.set(i, k, -V.get(i, k));
              }
            }
          }
          while (k < pp) {
            if (s[k] >= s[k + 1]) {
              break;
            }
            let t = s[k];
            s[k] = s[k + 1];
            s[k + 1] = t;
            if (wantv && k < n - 1) {
              for (let i = 0; i < n; i++) {
                t = V.get(i, k + 1);
                V.set(i, k + 1, V.get(i, k));
                V.set(i, k, t);
              }
            }
            if (wantu && k < m - 1) {
              for (let i = 0; i < m; i++) {
                t = U.get(i, k + 1);
                U.set(i, k + 1, U.get(i, k));
                U.set(i, k, t);
              }
            }
            k++;
          }
          p--;
          break;
        }
        // no default
      }
    }

    if (swapped) {
      let tmp = V;
      V = U;
      U = tmp;
    }

    this.m = m;
    this.n = n;
    this.s = s;
    this.U = U;
    this.V = V;
  }

  solve(value) {
    let Y = value;
    let e = this.threshold;
    let scols = this.s.length;
    let Ls = Matrix.zeros(scols, scols);

    for (let i = 0; i < scols; i++) {
      if (Math.abs(this.s[i]) <= e) {
        Ls.set(i, i, 0);
      } else {
        Ls.set(i, i, 1 / this.s[i]);
      }
    }

    let U = this.U;
    let V = this.rightSingularVectors;

    let VL = V.mmul(Ls);
    let vrows = V.rows;
    let urows = U.rows;
    let VLU = Matrix.zeros(vrows, urows);

    for (let i = 0; i < vrows; i++) {
      for (let j = 0; j < urows; j++) {
        let sum = 0;
        for (let k = 0; k < scols; k++) {
          sum += VL.get(i, k) * U.get(j, k);
        }
        VLU.set(i, j, sum);
      }
    }

    return VLU.mmul(Y);
  }

  solveForDiagonal(value) {
    return this.solve(Matrix.diag(value));
  }

  inverse() {
    let V = this.V;
    let e = this.threshold;
    let vrows = V.rows;
    let vcols = V.columns;
    let X = new Matrix(vrows, this.s.length);

    for (let i = 0; i < vrows; i++) {
      for (let j = 0; j < vcols; j++) {
        if (Math.abs(this.s[j]) > e) {
          X.set(i, j, V.get(i, j) / this.s[j]);
        }
      }
    }

    let U = this.U;

    let urows = U.rows;
    let ucols = U.columns;
    let Y = new Matrix(vrows, urows);

    for (let i = 0; i < vrows; i++) {
      for (let j = 0; j < urows; j++) {
        let sum = 0;
        for (let k = 0; k < ucols; k++) {
          sum += X.get(i, k) * U.get(j, k);
        }
        Y.set(i, j, sum);
      }
    }

    return Y;
  }

  get condition() {
    return this.s[0] / this.s[Math.min(this.m, this.n) - 1];
  }

  get norm2() {
    return this.s[0];
  }

  get rank() {
    let tol = Math.max(this.m, this.n) * this.s[0] * Number.EPSILON;
    let r = 0;
    let s = this.s;
    for (let i = 0, ii = s.length; i < ii; i++) {
      if (s[i] > tol) {
        r++;
      }
    }
    return r;
  }

  get diagonal() {
    return Array.from(this.s);
  }

  get threshold() {
    return (Number.EPSILON / 2) * Math.max(this.m, this.n) * this.s[0];
  }

  get leftSingularVectors() {
    return this.U;
  }

  get rightSingularVectors() {
    return this.V;
  }

  get diagonalMatrix() {
    return Matrix.diag(this.s);
  }
}

function inverse(matrix, useSVD = false) {
  matrix = WrapperMatrix2D.checkMatrix(matrix);
  if (useSVD) {
    return new SingularValueDecomposition(matrix).inverse();
  } else {
    return solve(matrix, Matrix.eye(matrix.rows));
  }
}

function solve(leftHandSide, rightHandSide, useSVD = false) {
  leftHandSide = WrapperMatrix2D.checkMatrix(leftHandSide);
  rightHandSide = WrapperMatrix2D.checkMatrix(rightHandSide);
  if (useSVD) {
    return new SingularValueDecomposition(leftHandSide).solve(rightHandSide);
  } else {
    return leftHandSide.isSquare()
      ? new LuDecomposition(leftHandSide).solve(rightHandSide)
      : new QrDecomposition(leftHandSide).solve(rightHandSide);
  }
}

function determinant(matrix) {
  matrix = Matrix.checkMatrix(matrix);
  if (matrix.isSquare()) {
    let a, b, c, d;
    if (matrix.columns === 2) {
      // 2 x 2 matrix
      a = matrix.get(0, 0);
      b = matrix.get(0, 1);
      c = matrix.get(1, 0);
      d = matrix.get(1, 1);

      return a * d - b * c;
    } else if (matrix.columns === 3) {
      // 3 x 3 matrix
      let subMatrix0, subMatrix1, subMatrix2;
      subMatrix0 = new MatrixSelectionView(matrix, [1, 2], [1, 2]);
      subMatrix1 = new MatrixSelectionView(matrix, [1, 2], [0, 2]);
      subMatrix2 = new MatrixSelectionView(matrix, [1, 2], [0, 1]);
      a = matrix.get(0, 0);
      b = matrix.get(0, 1);
      c = matrix.get(0, 2);

      return (
        a * determinant(subMatrix0) -
        b * determinant(subMatrix1) +
        c * determinant(subMatrix2)
      );
    } else {
      // general purpose determinant using the LU decomposition
      return new LuDecomposition(matrix).determinant;
    }
  } else {
    throw Error('determinant can only be calculated for a square matrix');
  }
}

function xrange(n, exception) {
  let range = [];
  for (let i = 0; i < n; i++) {
    if (i !== exception) {
      range.push(i);
    }
  }
  return range;
}

function dependenciesOneRow(
  error,
  matrix,
  index,
  thresholdValue = 10e-10,
  thresholdError = 10e-10,
) {
  if (error > thresholdError) {
    return new Array(matrix.rows + 1).fill(0);
  } else {
    let returnArray = matrix.addRow(index, [0]);
    for (let i = 0; i < returnArray.rows; i++) {
      if (Math.abs(returnArray.get(i, 0)) < thresholdValue) {
        returnArray.set(i, 0, 0);
      }
    }
    return returnArray.to1DArray();
  }
}

function linearDependencies(matrix, options = {}) {
  const { thresholdValue = 10e-10, thresholdError = 10e-10 } = options;
  matrix = Matrix.checkMatrix(matrix);

  let n = matrix.rows;
  let results = new Matrix(n, n);

  for (let i = 0; i < n; i++) {
    let b = Matrix.columnVector(matrix.getRow(i));
    let Abis = matrix.subMatrixRow(xrange(n, i)).transpose();
    let svd = new SingularValueDecomposition(Abis);
    let x = svd.solve(b);
    let error = Matrix.sub(b, Abis.mmul(x))
      .abs()
      .max();
    results.setRow(
      i,
      dependenciesOneRow(error, x, i, thresholdValue, thresholdError),
    );
  }
  return results;
}

function pseudoInverse(matrix, threshold = Number.EPSILON) {
  matrix = Matrix.checkMatrix(matrix);
  let svdSolution = new SingularValueDecomposition(matrix, { autoTranspose: true });

  let U = svdSolution.leftSingularVectors;
  let V = svdSolution.rightSingularVectors;
  let s = svdSolution.diagonal;

  for (let i = 0; i < s.length; i++) {
    if (Math.abs(s[i]) > threshold) {
      s[i] = 1.0 / s[i];
    } else {
      s[i] = 0.0;
    }
  }

  return V.mmul(Matrix.diag(s).mmul(U.transpose()));
}

function covariance(xMatrix, yMatrix = xMatrix, options = {}) {
  xMatrix = Matrix.checkMatrix(xMatrix);
  let yIsSame = false;
  if (
    typeof yMatrix === 'object' &&
    !Matrix.isMatrix(yMatrix) &&
    !Array.isArray(yMatrix)
  ) {
    options = yMatrix;
    yMatrix = xMatrix;
    yIsSame = true;
  } else {
    yMatrix = Matrix.checkMatrix(yMatrix);
  }
  if (xMatrix.rows !== yMatrix.rows) {
    throw new TypeError('Both matrices must have the same number of rows');
  }
  const { center = true } = options;
  if (center) {
    xMatrix = xMatrix.center('column');
    if (!yIsSame) {
      yMatrix = yMatrix.center('column');
    }
  }
  const cov = xMatrix.transpose().mmul(yMatrix);
  for (let i = 0; i < cov.rows; i++) {
    for (let j = 0; j < cov.columns; j++) {
      cov.set(i, j, cov.get(i, j) * (1 / (xMatrix.rows - 1)));
    }
  }
  return cov;
}

function correlation(xMatrix, yMatrix = xMatrix, options = {}) {
  xMatrix = Matrix.checkMatrix(xMatrix);
  let yIsSame = false;
  if (
    typeof yMatrix === 'object' &&
    !Matrix.isMatrix(yMatrix) &&
    !Array.isArray(yMatrix)
  ) {
    options = yMatrix;
    yMatrix = xMatrix;
    yIsSame = true;
  } else {
    yMatrix = Matrix.checkMatrix(yMatrix);
  }
  if (xMatrix.rows !== yMatrix.rows) {
    throw new TypeError('Both matrices must have the same number of rows');
  }

  const { center = true, scale = true } = options;
  if (center) {
    xMatrix.center('column');
    if (!yIsSame) {
      yMatrix.center('column');
    }
  }
  if (scale) {
    xMatrix.scale('column');
    if (!yIsSame) {
      yMatrix.scale('column');
    }
  }

  const sdx = xMatrix.standardDeviation('column', { unbiased: true });
  const sdy = yIsSame
    ? sdx
    : yMatrix.standardDeviation('column', { unbiased: true });

  const corr = xMatrix.transpose().mmul(yMatrix);
  for (let i = 0; i < corr.rows; i++) {
    for (let j = 0; j < corr.columns; j++) {
      corr.set(
        i,
        j,
        corr.get(i, j) * (1 / (sdx[i] * sdy[j])) * (1 / (xMatrix.rows - 1)),
      );
    }
  }
  return corr;
}

class EigenvalueDecomposition {
  constructor(matrix, options = {}) {
    const { assumeSymmetric = false } = options;

    matrix = WrapperMatrix2D.checkMatrix(matrix);
    if (!matrix.isSquare()) {
      throw new Error('Matrix is not a square matrix');
    }

    let n = matrix.columns;
    let V = new Matrix(n, n);
    let d = new Float64Array(n);
    let e = new Float64Array(n);
    let value = matrix;
    let i, j;

    let isSymmetric = false;
    if (assumeSymmetric) {
      isSymmetric = true;
    } else {
      isSymmetric = matrix.isSymmetric();
    }

    if (isSymmetric) {
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          V.set(i, j, value.get(i, j));
        }
      }
      tred2(n, e, d, V);
      tql2(n, e, d, V);
    } else {
      let H = new Matrix(n, n);
      let ort = new Float64Array(n);
      for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
          H.set(i, j, value.get(i, j));
        }
      }
      orthes(n, H, ort, V);
      hqr2(n, e, d, V, H);
    }

    this.n = n;
    this.e = e;
    this.d = d;
    this.V = V;
  }

  get realEigenvalues() {
    return Array.from(this.d);
  }

  get imaginaryEigenvalues() {
    return Array.from(this.e);
  }

  get eigenvectorMatrix() {
    return this.V;
  }

  get diagonalMatrix() {
    let n = this.n;
    let e = this.e;
    let d = this.d;
    let X = new Matrix(n, n);
    let i, j;
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        X.set(i, j, 0);
      }
      X.set(i, i, d[i]);
      if (e[i] > 0) {
        X.set(i, i + 1, e[i]);
      } else if (e[i] < 0) {
        X.set(i, i - 1, e[i]);
      }
    }
    return X;
  }
}

function tred2(n, e, d, V) {
  let f, g, h, i, j, k, hh, scale;

  for (j = 0; j < n; j++) {
    d[j] = V.get(n - 1, j);
  }

  for (i = n - 1; i > 0; i--) {
    scale = 0;
    h = 0;
    for (k = 0; k < i; k++) {
      scale = scale + Math.abs(d[k]);
    }

    if (scale === 0) {
      e[i] = d[i - 1];
      for (j = 0; j < i; j++) {
        d[j] = V.get(i - 1, j);
        V.set(i, j, 0);
        V.set(j, i, 0);
      }
    } else {
      for (k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }

      f = d[i - 1];
      g = Math.sqrt(h);
      if (f > 0) {
        g = -g;
      }

      e[i] = scale * g;
      h = h - f * g;
      d[i - 1] = f - g;
      for (j = 0; j < i; j++) {
        e[j] = 0;
      }

      for (j = 0; j < i; j++) {
        f = d[j];
        V.set(j, i, f);
        g = e[j] + V.get(j, j) * f;
        for (k = j + 1; k <= i - 1; k++) {
          g += V.get(k, j) * d[k];
          e[k] += V.get(k, j) * f;
        }
        e[j] = g;
      }

      f = 0;
      for (j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }

      hh = f / (h + h);
      for (j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }

      for (j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (k = j; k <= i - 1; k++) {
          V.set(k, j, V.get(k, j) - (f * e[k] + g * d[k]));
        }
        d[j] = V.get(i - 1, j);
        V.set(i, j, 0);
      }
    }
    d[i] = h;
  }

  for (i = 0; i < n - 1; i++) {
    V.set(n - 1, i, V.get(i, i));
    V.set(i, i, 1);
    h = d[i + 1];
    if (h !== 0) {
      for (k = 0; k <= i; k++) {
        d[k] = V.get(k, i + 1) / h;
      }

      for (j = 0; j <= i; j++) {
        g = 0;
        for (k = 0; k <= i; k++) {
          g += V.get(k, i + 1) * V.get(k, j);
        }
        for (k = 0; k <= i; k++) {
          V.set(k, j, V.get(k, j) - g * d[k]);
        }
      }
    }

    for (k = 0; k <= i; k++) {
      V.set(k, i + 1, 0);
    }
  }

  for (j = 0; j < n; j++) {
    d[j] = V.get(n - 1, j);
    V.set(n - 1, j, 0);
  }

  V.set(n - 1, n - 1, 1);
  e[0] = 0;
}

function tql2(n, e, d, V) {
  let g, h, i, j, k, l, m, p, r, dl1, c, c2, c3, el1, s, s2;

  for (i = 1; i < n; i++) {
    e[i - 1] = e[i];
  }

  e[n - 1] = 0;

  let f = 0;
  let tst1 = 0;
  let eps = Number.EPSILON;

  for (l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    m = l;
    while (m < n) {
      if (Math.abs(e[m]) <= eps * tst1) {
        break;
      }
      m++;
    }

    if (m > l) {
      do {

        g = d[l];
        p = (d[l + 1] - g) / (2 * e[l]);
        r = hypotenuse(p, 1);
        if (p < 0) {
          r = -r;
        }

        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        dl1 = d[l + 1];
        h = g - d[l];
        for (i = l + 2; i < n; i++) {
          d[i] -= h;
        }

        f = f + h;

        p = d[m];
        c = 1;
        c2 = c;
        c3 = c;
        el1 = e[l + 1];
        s = 0;
        s2 = 0;
        for (i = m - 1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypotenuse(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);

          for (k = 0; k < n; k++) {
            h = V.get(k, i + 1);
            V.set(k, i + 1, s * V.get(k, i) + c * h);
            V.set(k, i, c * V.get(k, i) - s * h);
          }
        }

        p = (-s * s2 * c3 * el1 * e[l]) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0;
  }

  for (i = 0; i < n - 1; i++) {
    k = i;
    p = d[i];
    for (j = i + 1; j < n; j++) {
      if (d[j] < p) {
        k = j;
        p = d[j];
      }
    }

    if (k !== i) {
      d[k] = d[i];
      d[i] = p;
      for (j = 0; j < n; j++) {
        p = V.get(j, i);
        V.set(j, i, V.get(j, k));
        V.set(j, k, p);
      }
    }
  }
}

function orthes(n, H, ort, V) {
  let low = 0;
  let high = n - 1;
  let f, g, h, i, j, m;
  let scale;

  for (m = low + 1; m <= high - 1; m++) {
    scale = 0;
    for (i = m; i <= high; i++) {
      scale = scale + Math.abs(H.get(i, m - 1));
    }

    if (scale !== 0) {
      h = 0;
      for (i = high; i >= m; i--) {
        ort[i] = H.get(i, m - 1) / scale;
        h += ort[i] * ort[i];
      }

      g = Math.sqrt(h);
      if (ort[m] > 0) {
        g = -g;
      }

      h = h - ort[m] * g;
      ort[m] = ort[m] - g;

      for (j = m; j < n; j++) {
        f = 0;
        for (i = high; i >= m; i--) {
          f += ort[i] * H.get(i, j);
        }

        f = f / h;
        for (i = m; i <= high; i++) {
          H.set(i, j, H.get(i, j) - f * ort[i]);
        }
      }

      for (i = 0; i <= high; i++) {
        f = 0;
        for (j = high; j >= m; j--) {
          f += ort[j] * H.get(i, j);
        }

        f = f / h;
        for (j = m; j <= high; j++) {
          H.set(i, j, H.get(i, j) - f * ort[j]);
        }
      }

      ort[m] = scale * ort[m];
      H.set(m, m - 1, scale * g);
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      V.set(i, j, i === j ? 1 : 0);
    }
  }

  for (m = high - 1; m >= low + 1; m--) {
    if (H.get(m, m - 1) !== 0) {
      for (i = m + 1; i <= high; i++) {
        ort[i] = H.get(i, m - 1);
      }

      for (j = m; j <= high; j++) {
        g = 0;
        for (i = m; i <= high; i++) {
          g += ort[i] * V.get(i, j);
        }

        g = g / ort[m] / H.get(m, m - 1);
        for (i = m; i <= high; i++) {
          V.set(i, j, V.get(i, j) + g * ort[i]);
        }
      }
    }
  }
}

function hqr2(nn, e, d, V, H) {
  let n = nn - 1;
  let low = 0;
  let high = nn - 1;
  let eps = Number.EPSILON;
  let exshift = 0;
  let norm = 0;
  let p = 0;
  let q = 0;
  let r = 0;
  let s = 0;
  let z = 0;
  let iter = 0;
  let i, j, k, l, m, t, w, x, y;
  let ra, sa, vr, vi;
  let notlast, cdivres;

  for (i = 0; i < nn; i++) {
    if (i < low || i > high) {
      d[i] = H.get(i, i);
      e[i] = 0;
    }

    for (j = Math.max(i - 1, 0); j < nn; j++) {
      norm = norm + Math.abs(H.get(i, j));
    }
  }

  while (n >= low) {
    l = n;
    while (l > low) {
      s = Math.abs(H.get(l - 1, l - 1)) + Math.abs(H.get(l, l));
      if (s === 0) {
        s = norm;
      }
      if (Math.abs(H.get(l, l - 1)) < eps * s) {
        break;
      }
      l--;
    }

    if (l === n) {
      H.set(n, n, H.get(n, n) + exshift);
      d[n] = H.get(n, n);
      e[n] = 0;
      n--;
      iter = 0;
    } else if (l === n - 1) {
      w = H.get(n, n - 1) * H.get(n - 1, n);
      p = (H.get(n - 1, n - 1) - H.get(n, n)) / 2;
      q = p * p + w;
      z = Math.sqrt(Math.abs(q));
      H.set(n, n, H.get(n, n) + exshift);
      H.set(n - 1, n - 1, H.get(n - 1, n - 1) + exshift);
      x = H.get(n, n);

      if (q >= 0) {
        z = p >= 0 ? p + z : p - z;
        d[n - 1] = x + z;
        d[n] = d[n - 1];
        if (z !== 0) {
          d[n] = x - w / z;
        }
        e[n - 1] = 0;
        e[n] = 0;
        x = H.get(n, n - 1);
        s = Math.abs(x) + Math.abs(z);
        p = x / s;
        q = z / s;
        r = Math.sqrt(p * p + q * q);
        p = p / r;
        q = q / r;

        for (j = n - 1; j < nn; j++) {
          z = H.get(n - 1, j);
          H.set(n - 1, j, q * z + p * H.get(n, j));
          H.set(n, j, q * H.get(n, j) - p * z);
        }

        for (i = 0; i <= n; i++) {
          z = H.get(i, n - 1);
          H.set(i, n - 1, q * z + p * H.get(i, n));
          H.set(i, n, q * H.get(i, n) - p * z);
        }

        for (i = low; i <= high; i++) {
          z = V.get(i, n - 1);
          V.set(i, n - 1, q * z + p * V.get(i, n));
          V.set(i, n, q * V.get(i, n) - p * z);
        }
      } else {
        d[n - 1] = x + p;
        d[n] = x + p;
        e[n - 1] = z;
        e[n] = -z;
      }

      n = n - 2;
      iter = 0;
    } else {
      x = H.get(n, n);
      y = 0;
      w = 0;
      if (l < n) {
        y = H.get(n - 1, n - 1);
        w = H.get(n, n - 1) * H.get(n - 1, n);
      }

      if (iter === 10) {
        exshift += x;
        for (i = low; i <= n; i++) {
          H.set(i, i, H.get(i, i) - x);
        }
        s = Math.abs(H.get(n, n - 1)) + Math.abs(H.get(n - 1, n - 2));
        x = y = 0.75 * s;
        w = -0.4375 * s * s;
      }

      if (iter === 30) {
        s = (y - x) / 2;
        s = s * s + w;
        if (s > 0) {
          s = Math.sqrt(s);
          if (y < x) {
            s = -s;
          }
          s = x - w / ((y - x) / 2 + s);
          for (i = low; i <= n; i++) {
            H.set(i, i, H.get(i, i) - s);
          }
          exshift += s;
          x = y = w = 0.964;
        }
      }

      iter = iter + 1;

      m = n - 2;
      while (m >= l) {
        z = H.get(m, m);
        r = x - z;
        s = y - z;
        p = (r * s - w) / H.get(m + 1, m) + H.get(m, m + 1);
        q = H.get(m + 1, m + 1) - z - r - s;
        r = H.get(m + 2, m + 1);
        s = Math.abs(p) + Math.abs(q) + Math.abs(r);
        p = p / s;
        q = q / s;
        r = r / s;
        if (m === l) {
          break;
        }
        if (
          Math.abs(H.get(m, m - 1)) * (Math.abs(q) + Math.abs(r)) <
          eps *
            (Math.abs(p) *
              (Math.abs(H.get(m - 1, m - 1)) +
                Math.abs(z) +
                Math.abs(H.get(m + 1, m + 1))))
        ) {
          break;
        }
        m--;
      }

      for (i = m + 2; i <= n; i++) {
        H.set(i, i - 2, 0);
        if (i > m + 2) {
          H.set(i, i - 3, 0);
        }
      }

      for (k = m; k <= n - 1; k++) {
        notlast = k !== n - 1;
        if (k !== m) {
          p = H.get(k, k - 1);
          q = H.get(k + 1, k - 1);
          r = notlast ? H.get(k + 2, k - 1) : 0;
          x = Math.abs(p) + Math.abs(q) + Math.abs(r);
          if (x !== 0) {
            p = p / x;
            q = q / x;
            r = r / x;
          }
        }

        if (x === 0) {
          break;
        }

        s = Math.sqrt(p * p + q * q + r * r);
        if (p < 0) {
          s = -s;
        }

        if (s !== 0) {
          if (k !== m) {
            H.set(k, k - 1, -s * x);
          } else if (l !== m) {
            H.set(k, k - 1, -H.get(k, k - 1));
          }

          p = p + s;
          x = p / s;
          y = q / s;
          z = r / s;
          q = q / p;
          r = r / p;

          for (j = k; j < nn; j++) {
            p = H.get(k, j) + q * H.get(k + 1, j);
            if (notlast) {
              p = p + r * H.get(k + 2, j);
              H.set(k + 2, j, H.get(k + 2, j) - p * z);
            }

            H.set(k, j, H.get(k, j) - p * x);
            H.set(k + 1, j, H.get(k + 1, j) - p * y);
          }

          for (i = 0; i <= Math.min(n, k + 3); i++) {
            p = x * H.get(i, k) + y * H.get(i, k + 1);
            if (notlast) {
              p = p + z * H.get(i, k + 2);
              H.set(i, k + 2, H.get(i, k + 2) - p * r);
            }

            H.set(i, k, H.get(i, k) - p);
            H.set(i, k + 1, H.get(i, k + 1) - p * q);
          }

          for (i = low; i <= high; i++) {
            p = x * V.get(i, k) + y * V.get(i, k + 1);
            if (notlast) {
              p = p + z * V.get(i, k + 2);
              V.set(i, k + 2, V.get(i, k + 2) - p * r);
            }

            V.set(i, k, V.get(i, k) - p);
            V.set(i, k + 1, V.get(i, k + 1) - p * q);
          }
        }
      }
    }
  }

  if (norm === 0) {
    return;
  }

  for (n = nn - 1; n >= 0; n--) {
    p = d[n];
    q = e[n];

    if (q === 0) {
      l = n;
      H.set(n, n, 1);
      for (i = n - 1; i >= 0; i--) {
        w = H.get(i, i) - p;
        r = 0;
        for (j = l; j <= n; j++) {
          r = r + H.get(i, j) * H.get(j, n);
        }

        if (e[i] < 0) {
          z = w;
          s = r;
        } else {
          l = i;
          if (e[i] === 0) {
            H.set(i, n, w !== 0 ? -r / w : -r / (eps * norm));
          } else {
            x = H.get(i, i + 1);
            y = H.get(i + 1, i);
            q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
            t = (x * s - z * r) / q;
            H.set(i, n, t);
            H.set(
              i + 1,
              n,
              Math.abs(x) > Math.abs(z) ? (-r - w * t) / x : (-s - y * t) / z,
            );
          }

          t = Math.abs(H.get(i, n));
          if (eps * t * t > 1) {
            for (j = i; j <= n; j++) {
              H.set(j, n, H.get(j, n) / t);
            }
          }
        }
      }
    } else if (q < 0) {
      l = n - 1;

      if (Math.abs(H.get(n, n - 1)) > Math.abs(H.get(n - 1, n))) {
        H.set(n - 1, n - 1, q / H.get(n, n - 1));
        H.set(n - 1, n, -(H.get(n, n) - p) / H.get(n, n - 1));
      } else {
        cdivres = cdiv(0, -H.get(n - 1, n), H.get(n - 1, n - 1) - p, q);
        H.set(n - 1, n - 1, cdivres[0]);
        H.set(n - 1, n, cdivres[1]);
      }

      H.set(n, n - 1, 0);
      H.set(n, n, 1);
      for (i = n - 2; i >= 0; i--) {
        ra = 0;
        sa = 0;
        for (j = l; j <= n; j++) {
          ra = ra + H.get(i, j) * H.get(j, n - 1);
          sa = sa + H.get(i, j) * H.get(j, n);
        }

        w = H.get(i, i) - p;

        if (e[i] < 0) {
          z = w;
          r = ra;
          s = sa;
        } else {
          l = i;
          if (e[i] === 0) {
            cdivres = cdiv(-ra, -sa, w, q);
            H.set(i, n - 1, cdivres[0]);
            H.set(i, n, cdivres[1]);
          } else {
            x = H.get(i, i + 1);
            y = H.get(i + 1, i);
            vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
            vi = (d[i] - p) * 2 * q;
            if (vr === 0 && vi === 0) {
              vr =
                eps *
                norm *
                (Math.abs(w) +
                  Math.abs(q) +
                  Math.abs(x) +
                  Math.abs(y) +
                  Math.abs(z));
            }
            cdivres = cdiv(
              x * r - z * ra + q * sa,
              x * s - z * sa - q * ra,
              vr,
              vi,
            );
            H.set(i, n - 1, cdivres[0]);
            H.set(i, n, cdivres[1]);
            if (Math.abs(x) > Math.abs(z) + Math.abs(q)) {
              H.set(
                i + 1,
                n - 1,
                (-ra - w * H.get(i, n - 1) + q * H.get(i, n)) / x,
              );
              H.set(
                i + 1,
                n,
                (-sa - w * H.get(i, n) - q * H.get(i, n - 1)) / x,
              );
            } else {
              cdivres = cdiv(
                -r - y * H.get(i, n - 1),
                -s - y * H.get(i, n),
                z,
                q,
              );
              H.set(i + 1, n - 1, cdivres[0]);
              H.set(i + 1, n, cdivres[1]);
            }
          }

          t = Math.max(Math.abs(H.get(i, n - 1)), Math.abs(H.get(i, n)));
          if (eps * t * t > 1) {
            for (j = i; j <= n; j++) {
              H.set(j, n - 1, H.get(j, n - 1) / t);
              H.set(j, n, H.get(j, n) / t);
            }
          }
        }
      }
    }
  }

  for (i = 0; i < nn; i++) {
    if (i < low || i > high) {
      for (j = i; j < nn; j++) {
        V.set(i, j, H.get(i, j));
      }
    }
  }

  for (j = nn - 1; j >= low; j--) {
    for (i = low; i <= high; i++) {
      z = 0;
      for (k = low; k <= Math.min(j, high); k++) {
        z = z + V.get(i, k) * H.get(k, j);
      }
      V.set(i, j, z);
    }
  }
}

function cdiv(xr, xi, yr, yi) {
  let r, d;
  if (Math.abs(yr) > Math.abs(yi)) {
    r = yi / yr;
    d = yr + r * yi;
    return [(xr + r * xi) / d, (xi - r * xr) / d];
  } else {
    r = yr / yi;
    d = yi + r * yr;
    return [(r * xr + xi) / d, (r * xi - xr) / d];
  }
}

class CholeskyDecomposition {
  constructor(value) {
    value = WrapperMatrix2D.checkMatrix(value);
    if (!value.isSymmetric()) {
      throw new Error('Matrix is not symmetric');
    }

    let a = value;
    let dimension = a.rows;
    let l = new Matrix(dimension, dimension);
    let positiveDefinite = true;
    let i, j, k;

    for (j = 0; j < dimension; j++) {
      let d = 0;
      for (k = 0; k < j; k++) {
        let s = 0;
        for (i = 0; i < k; i++) {
          s += l.get(k, i) * l.get(j, i);
        }
        s = (a.get(j, k) - s) / l.get(k, k);
        l.set(j, k, s);
        d = d + s * s;
      }

      d = a.get(j, j) - d;

      positiveDefinite &= d > 0;
      l.set(j, j, Math.sqrt(Math.max(d, 0)));
      for (k = j + 1; k < dimension; k++) {
        l.set(j, k, 0);
      }
    }

    this.L = l;
    this.positiveDefinite = Boolean(positiveDefinite);
  }

  isPositiveDefinite() {
    return this.positiveDefinite;
  }

  solve(value) {
    value = WrapperMatrix2D.checkMatrix(value);

    let l = this.L;
    let dimension = l.rows;

    if (value.rows !== dimension) {
      throw new Error('Matrix dimensions do not match');
    }
    if (this.isPositiveDefinite() === false) {
      throw new Error('Matrix is not positive definite');
    }

    let count = value.columns;
    let B = value.clone();
    let i, j, k;

    for (k = 0; k < dimension; k++) {
      for (j = 0; j < count; j++) {
        for (i = 0; i < k; i++) {
          B.set(k, j, B.get(k, j) - B.get(i, j) * l.get(k, i));
        }
        B.set(k, j, B.get(k, j) / l.get(k, k));
      }
    }

    for (k = dimension - 1; k >= 0; k--) {
      for (j = 0; j < count; j++) {
        for (i = k + 1; i < dimension; i++) {
          B.set(k, j, B.get(k, j) - B.get(i, j) * l.get(i, k));
        }
        B.set(k, j, B.get(k, j) / l.get(k, k));
      }
    }

    return B;
  }

  get lowerTriangularMatrix() {
    return this.L;
  }
}

class nipals {
  constructor(X, options = {}) {
    X = WrapperMatrix2D.checkMatrix(X);
    let { Y } = options;
    const {
      scaleScores = false,
      maxIterations = 1000,
      terminationCriteria = 1e-10,
    } = options;

    let u;
    if (Y) {
      if (Array.isArray(Y) && typeof Y[0] === 'number') {
        Y = Matrix.columnVector(Y);
      } else {
        Y = WrapperMatrix2D.checkMatrix(Y);
      }
      if (!Y.isColumnVector() || Y.rows !== X.rows) {
        throw new Error('Y must be a column vector of length X.rows');
      }
      u = Y;
    } else {
      u = X.getColumnVector(0);
    }

    let diff = 1;
    let t, q, w, tOld;

    for (
      let counter = 0;
      counter < maxIterations && diff > terminationCriteria;
      counter++
    ) {
      w = X.transpose()
        .mmul(u)
        .div(
          u
            .transpose()
            .mmul(u)
            .get(0, 0),
        );
      w = w.div(w.norm());

      t = X.mmul(w).div(
        w
          .transpose()
          .mmul(w)
          .get(0, 0),
      );

      if (counter > 0) {
        diff = t
          .clone()
          .sub(tOld)
          .pow(2)
          .sum();
      }
      tOld = t.clone();

      if (Y) {
        q = Y.transpose()
          .mmul(t)
          .div(
            t
              .transpose()
              .mmul(t)
              .get(0, 0),
          );
        q = q.div(q.norm());

        u = Y.mmul(q).div(
          q
            .transpose()
            .mmul(q)
            .get(0, 0),
        );
      } else {
        u = t;
      }
    }

    if (Y) {
      let p = X.transpose()
        .mmul(t)
        .div(
          t
            .transpose()
            .mmul(t)
            .get(0, 0),
        );
      p = p.div(p.norm());
      let xResidual = X.clone().sub(t.clone().mmul(p.transpose()));
      let residual = u
        .transpose()
        .mmul(t)
        .div(
          t
            .transpose()
            .mmul(t)
            .get(0, 0),
        );
      let yResidual = Y.clone().sub(
        t
          .clone()
          .mulS(residual.get(0, 0))
          .mmul(q.transpose()),
      );

      this.t = t;
      this.p = p.transpose();
      this.w = w.transpose();
      this.q = q;
      this.u = u;
      this.s = t.transpose().mmul(t);
      this.xResidual = xResidual;
      this.yResidual = yResidual;
      this.betas = residual;
    } else {
      this.w = w.transpose();
      this.s = t
        .transpose()
        .mmul(t)
        .sqrt();
      if (scaleScores) {
        this.t = t.clone().div(this.s.get(0, 0));
      } else {
        this.t = t;
      }
      this.xResidual = X.sub(t.mmul(w.transpose()));
    }
  }
}

exports.AbstractMatrix = AbstractMatrix;
exports.CHO = CholeskyDecomposition;
exports.CholeskyDecomposition = CholeskyDecomposition;
exports.EVD = EigenvalueDecomposition;
exports.EigenvalueDecomposition = EigenvalueDecomposition;
exports.LU = LuDecomposition;
exports.LuDecomposition = LuDecomposition;
exports.Matrix = Matrix;
exports.MatrixColumnSelectionView = MatrixColumnSelectionView;
exports.MatrixColumnView = MatrixColumnView;
exports.MatrixFlipColumnView = MatrixFlipColumnView;
exports.MatrixFlipRowView = MatrixFlipRowView;
exports.MatrixRowSelectionView = MatrixRowSelectionView;
exports.MatrixRowView = MatrixRowView;
exports.MatrixSelectionView = MatrixSelectionView;
exports.MatrixSubView = MatrixSubView;
exports.MatrixTransposeView = MatrixTransposeView;
exports.NIPALS = nipals;
exports.Nipals = nipals;
exports.QR = QrDecomposition;
exports.QrDecomposition = QrDecomposition;
exports.SVD = SingularValueDecomposition;
exports.SingularValueDecomposition = SingularValueDecomposition;
exports.WrapperMatrix1D = WrapperMatrix1D;
exports.WrapperMatrix2D = WrapperMatrix2D;
exports.correlation = correlation;
exports.covariance = covariance;
exports.default = Matrix;
exports.determinant = determinant;
exports.inverse = inverse;
exports.linearDependencies = linearDependencies;
exports.pseudoInverse = pseudoInverse;
exports.solve = solve;
exports.wrap = wrap;

},{"ml-array-rescale":6}],8:[function(require,module,exports){
'use strict'

const symbols = require('./lib/symbols')
const number = require('./lib/number')
const string = require('./lib/string')
const object = require('./lib/object')

const typePredicates = {
  number,
  string,
  object
}

const createOw = ({
  validators = [],
  predicates = typePredicates,
  type = null
} = { }) => {
  const ow = new Proxy(function () { }, {
    get: (obj, key) => {
      if (key === symbols.validate) {
        return (value, label = 'argument') => {
          if (!type) {
            return new Error('missing required type specifier')
          }

          for (let i = 0; i < validators.length; ++i) {
            const validator = validators[i]
            const result = validator.fn(value)

            if (!result) {
              if (i === 0) {
                throw new Error(`Expected ${label} \`${value}\` to be of type \`${type}\`, but received type \`${typeof value}\``)
              } else {
                throw new Error(`Expected ${type} \`${label}\` \`${value}\` failed predicate \`${validator.key}\``)
              }
            }
          }
        }
      }

      const predicate = predicates[key]

      if (predicate) {
        if (typeof predicate === 'function') {
          validators.push({
            key,
            fn: predicate
          })

          return ow
        } else {
          return createOw({
            type: key,
            validators: [
              {
                key,
                fn: predicate.validator
              }
            ],
            predicates: predicate.predicates
          })
        }
      } else {
        const fn = predicates[symbols.func] && predicates[symbols.func][key]

        if (fn) {
          return new Proxy(function () { }, {
            get: () => {
              throw new Error(`invalid use of functional predicate "${key}"`)
            },

            apply: (obj, thisArg, args) => {
              validators.push({
                key,
                fn: fn(...args)
              })

              return ow
            }
          })
        } else {
          if (key === 'default' || key === '__esModule') {
            return ow
          }

          return ow
          // throw new Error(`unrecognized predicate "${key}"`)
        }
      }
    },

    apply: (obj, thisArg, args) => {
      if (args.length !== 2 && args.length !== 3) {
        throw new Error('invalid number of arguments to "ow"')
      } else {
        args[1][symbols.validate](args[0], args[2])
      }
    }
  })

  return ow
}

module.exports = createOw()

},{"./lib/number":9,"./lib/object":10,"./lib/string":11,"./lib/symbols":12}],9:[function(require,module,exports){
'use strict'

const { func } = require('./symbols')

const numberPredicates = {
  positive: (value) => (value > 0),
  negative: (value) => (value < 0),
  nonNegative: (value) => (value >= 0),
  integer: (value) => (value === (value | 0)),

  [func]: {
    is: (fn) => fn,
    eq: (v) => (value) => (value === v),
    gt: (v) => (value) => (value > v),
    gte: (v) => (value) => (value >= v),
    lt: (v) => (value) => (value < v),
    lte: (v) => (value) => (value <= v)
  }
}

module.exports = {
  predicates: numberPredicates,
  validator: (value) => {
    return typeof value === 'number'
  }
}

},{"./symbols":12}],10:[function(require,module,exports){
'use strict'

const { func } = require('./symbols')

const objectPredicates = {
  plain: (value) => {
    if (typeof value !== 'object') return false

    const proto = Object.getPrototypeOf(value)
    return proto === null || proto === Object.getPrototypeOf({})
  },
  empty: (value) => Object.keys(value).length === 0,
  nonEmpty: (value) => Object.keys(value).length > 0,

  [func]: {
    is: (fn) => fn,
    instanceOf: (v) => (value) => (value instanceof v)
  }
}

module.exports = {
  predicates: objectPredicates,
  validator: (value) => {
    return typeof value === 'object'
  }
}

},{"./symbols":12}],11:[function(require,module,exports){
'use strict'

const { func } = require('./symbols')

const stringPredicates = {
  empty: (value) => (value === ''),
  nonEmpty: (value) => (value !== ''),

  [func]: {
    is: (fn) => fn,
    eq: (v) => (value) => (value === v),
    length: (v) => (value) => (value.length === v),
    minLength: (v) => (value) => (value.length >= v),
    maxLength: (v) => (value) => (value.length <= v),
    matches: (v) => (value) => v.test(value),
    startsWith: (v) => (value) => value.startsWith(v),
    endsWith: (v) => (value) => value.endsWith(v)
  }
}

module.exports = {
  predicates: stringPredicates,
  validator: (value) => {
    return typeof value === 'string'
  }
}

},{"./symbols":12}],12:[function(require,module,exports){
'use strict'

exports.func = Symbol('func')
exports.validate = Symbol('validate')

},{}],13:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random) {
  var n = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;

  (0, _owLite2.default)(n, _owLite2.default.number.integer.positive);
  var irwinHall = random.irwinHall(n);

  return function () {
    return irwinHall() / n;
  };
};

},{"ow-lite":8}],14:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random) {
  var p = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0.5;

  (0, _owLite2.default)(p, _owLite2.default.number.gte(0).lt(1));

  return function () {
    return random.next() + p | 0;
  };
};

},{"ow-lite":8}],15:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random) {
  var n = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;
  var p = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 0.5;

  (0, _owLite2.default)(n, _owLite2.default.number.positive.integer);
  (0, _owLite2.default)(p, _owLite2.default.number.gte(0).lte(1));

  return function () {
    var i = 0;
    var x = 0;

    while (i++ < n) {
      x += random.next() < p;
    }

    return x;
  };
};

},{"ow-lite":8}],16:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random) {
  var lambda = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;

  (0, _owLite2.default)(lambda, _owLite2.default.number.positive);

  return function () {
    return -Math.log(1 - random.next()) / lambda;
  };
};

},{"ow-lite":8}],17:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random) {
  var p = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0.5;

  (0, _owLite2.default)(p, _owLite2.default.number.gt(0).lte(1));
  var invLogP = 1.0 / Math.log(1.0 - p);

  return function () {
    return 1 + Math.log(random.next()) * invLogP | 0;
  };
};

},{"ow-lite":8}],18:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random) {
  var n = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;

  (0, _owLite2.default)(n, _owLite2.default.number.integer.gte(0));

  return function () {
    var sum = 0;
    for (var i = 0; i < n; ++i) {
      sum += random.next();
    }

    return sum;
  };
};

},{"ow-lite":8}],19:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});

exports.default = function (random) {
  for (var _len = arguments.length, args = Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    args[_key - 1] = arguments[_key];
  }

  var normal = random.normal.apply(random, args);

  return function () {
    return Math.exp(normal());
  };
};

},{}],20:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random) {
  var mu = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0;
  var sigma = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 1;

  (0, _owLite2.default)(mu, _owLite2.default.number);
  (0, _owLite2.default)(sigma, _owLite2.default.number);

  return function () {
    var x = void 0,
        y = void 0,
        r = void 0;

    do {
      x = random.next() * 2 - 1;
      y = random.next() * 2 - 1;
      r = x * x + y * y;
    } while (!r || r > 1);

    return mu + sigma * y * Math.sqrt(-2 * Math.log(r) / r);
  };
};

},{"ow-lite":8}],21:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random) {
  var alpha = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;

  (0, _owLite2.default)(alpha, _owLite2.default.number.gte(0));
  var invAlpha = 1.0 / alpha;

  return function () {
    return 1.0 / Math.pow(1.0 - random.next(), invAlpha);
  };
};

},{"ow-lite":8}],22:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var logFactorialTable = [0.0, 0.0, 0.69314718055994529, 1.7917594692280550, 3.1780538303479458, 4.7874917427820458, 6.5792512120101012, 8.5251613610654147, 10.604602902745251, 12.801827480081469];

var logFactorial = function logFactorial(k) {
  return logFactorialTable[k];
};

var logSqrt2PI = 0.91893853320467267;

exports.default = function (random) {
  var lambda = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;

  (0, _owLite2.default)(lambda, _owLite2.default.number.positive);

  if (lambda < 10) {
    // inversion method
    var expMean = Math.exp(-lambda);

    return function () {
      var p = expMean;
      var x = 0;
      var u = random.next();

      while (u > p) {
        u = u - p;
        p = lambda * p / ++x;
      }

      return x;
    };
  } else {
    // generative method
    var smu = Math.sqrt(lambda);
    var b = 0.931 + 2.53 * smu;
    var a = -0.059 + 0.02483 * b;
    var invAlpha = 1.1239 + 1.1328 / (b - 3.4);
    var vR = 0.9277 - 3.6224 / (b - 2);

    return function () {
      while (true) {
        var u = void 0;
        var v = random.next();

        if (v <= 0.86 * vR) {
          u = v / vR - 0.43;
          return Math.floor((2 * a / (0.5 - Math.abs(u)) + b) * u + lambda + 0.445);
        }

        if (v >= vR) {
          u = random.next() - 0.5;
        } else {
          u = v / vR - 0.93;
          u = (u < 0 ? -0.5 : 0.5) - u;
          v = random.next() * vR;
        }

        var us = 0.5 - Math.abs(u);
        if (us < 0.013 && v > us) {
          continue;
        }

        var k = Math.floor((2 * a / us + b) * u + lambda + 0.445) | 0;
        v = v * invAlpha / (a / (us * us) + b);

        if (k >= 10) {
          var t = (k + 0.5) * Math.log(lambda / k) - lambda - logSqrt2PI + k - (1 / 12.0 - (1 / 360.0 - 1 / (1260.0 * k * k)) / (k * k)) / k;

          if (Math.log(v * smu) <= t) {
            return k;
          }
        } else if (k >= 0) {
          if (Math.log(v) <= k * Math.log(lambda) - lambda - logFactorial(k)) {
            return k;
          }
        }
      }
    };
  }
};

},{"ow-lite":8}],23:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});

exports.default = function (random) {
  return function () {
    return random.next() >= 0.5;
  };
};

},{}],24:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random, min, max) {
  if (max === undefined) {
    max = min === undefined ? 1 : min;
    min = 0;
  }

  (0, _owLite2.default)(min, _owLite2.default.number.integer);
  (0, _owLite2.default)(max, _owLite2.default.number.integer);

  return function () {
    return random.next() * (max - min + 1) + min | 0;
  };
};

},{"ow-lite":8}],25:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = function (random, min, max) {
  if (max === undefined) {
    max = min === undefined ? 1 : min;
    min = 0;
  }

  (0, _owLite2.default)(min, _owLite2.default.number);
  (0, _owLite2.default)(max, _owLite2.default.number);

  return function () {
    return random.next() * (max - min) + min;
  };
};

},{"ow-lite":8}],26:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

var _rng = require('../rng');

var _rng2 = _interopRequireDefault(_rng);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var RNGFunction = function (_RNG) {
  _inherits(RNGFunction, _RNG);

  function RNGFunction(thunk, opts) {
    _classCallCheck(this, RNGFunction);

    var _this = _possibleConstructorReturn(this, (RNGFunction.__proto__ || Object.getPrototypeOf(RNGFunction)).call(this));

    _this.seed(thunk, opts);
    return _this;
  }

  _createClass(RNGFunction, [{
    key: 'next',
    value: function next() {
      return this._rng();
    }
  }, {
    key: 'seed',
    value: function seed(thunk) {
      (0, _owLite2.default)(thunk, _owLite2.default.function);
      this._rng = thunk;
    }
  }, {
    key: 'clone',
    value: function clone() {
      for (var _len = arguments.length, opts = Array(_len), _key = 0; _key < _len; _key++) {
        opts[_key] = arguments[_key];
      }

      return new (Function.prototype.bind.apply(RNGFunction, [null].concat([this._rng], opts)))();
    }
  }, {
    key: 'name',
    get: function get() {
      return 'function';
    }
  }]);

  return RNGFunction;
}(_rng2.default);

exports.default = RNGFunction;

},{"../rng":29,"ow-lite":8}],27:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.RNGFactory = exports.RNG = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _owLite = require('ow-lite');

var _owLite2 = _interopRequireDefault(_owLite);

var _rng = require('./rng');

var _rng2 = _interopRequireDefault(_rng);

var _rngFactory = require('./rng-factory');

var _rngFactory2 = _interopRequireDefault(_rngFactory);

var _uniform2 = require('./distributions/uniform');

var _uniform3 = _interopRequireDefault(_uniform2);

var _uniformInt2 = require('./distributions/uniform-int');

var _uniformInt3 = _interopRequireDefault(_uniformInt2);

var _uniformBoolean2 = require('./distributions/uniform-boolean');

var _uniformBoolean3 = _interopRequireDefault(_uniformBoolean2);

var _normal2 = require('./distributions/normal');

var _normal3 = _interopRequireDefault(_normal2);

var _logNormal2 = require('./distributions/log-normal');

var _logNormal3 = _interopRequireDefault(_logNormal2);

var _bernoulli2 = require('./distributions/bernoulli');

var _bernoulli3 = _interopRequireDefault(_bernoulli2);

var _binomial2 = require('./distributions/binomial');

var _binomial3 = _interopRequireDefault(_binomial2);

var _geometric2 = require('./distributions/geometric');

var _geometric3 = _interopRequireDefault(_geometric2);

var _poisson2 = require('./distributions/poisson');

var _poisson3 = _interopRequireDefault(_poisson2);

var _exponential2 = require('./distributions/exponential');

var _exponential3 = _interopRequireDefault(_exponential2);

var _irwinHall2 = require('./distributions/irwin-hall');

var _irwinHall3 = _interopRequireDefault(_irwinHall2);

var _bates2 = require('./distributions/bates');

var _bates3 = _interopRequireDefault(_bates2);

var _pareto2 = require('./distributions/pareto');

var _pareto3 = _interopRequireDefault(_pareto2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

exports.RNG = _rng2.default;
exports.RNGFactory = _rngFactory2.default;

/**
 * Seedable random number generator supporting many common distributions.
 *
 * Defaults to Math.random as its underlying pseudorandom number generator.
 *
 * @name Random
 * @class
 *
 * @param {RNG|function} [rng=Math.random] - Underlying pseudorandom number generator.
 */

var Random = function () {
  function Random(rng) {
    _classCallCheck(this, Random);

    if (rng) (0, _owLite2.default)(rng, _owLite2.default.object.instanceOf(_rng2.default));

    this._cache = {};
    this.use(rng);
  }

  /**
   * @member {RNG} Underlying pseudo-random number generator
   */


  _createClass(Random, [{
    key: 'clone',


    /**
     * Creates a new `Random` instance, optionally specifying parameters to
     * set a new seed.
     *
     * @see RNG.clone
     *
     * @param {string} [seed] - Optional seed for new RNG.
     * @param {object} [opts] - Optional config for new RNG options.
     * @return {Random}
     */
    value: function clone() {
      if (arguments.length) {
        return new Random(_rngFactory2.default.apply(undefined, arguments));
      } else {
        return new Random(this.rng.clone());
      }
    }

    /**
     * Sets the underlying pseudorandom number generator used via
     * either an instance of `seedrandom`, a custom instance of RNG
     * (for PRNG plugins), or a string specifying the PRNG to use
     * along with an optional `seed` and `opts` to initialize the
     * RNG.
     *
     * @example
     * const random = require('random')
     *
     * random.use('example_seedrandom_string')
     * // or
     * random.use(seedrandom('kittens'))
     * // or
     * random.use(Math.random)
     *
     * @param {...*} args
     */

  }, {
    key: 'use',
    value: function use() {
      this._rng = _rngFactory2.default.apply(undefined, arguments);
    }

    /**
     * Patches `Math.random` with this Random instance's PRNG.
     */

  }, {
    key: 'patch',
    value: function patch() {
      if (this._patch) {
        throw new Error('Math.random already patched');
      }

      this._patch = Math.random;
      Math.random = this.uniform();
    }

    /**
     * Restores a previously patched `Math.random` to its original value.
     */

  }, {
    key: 'unpatch',
    value: function unpatch() {
      if (this._patch) {
        Math.random = this._patch;
        delete this._patch;
      }
    }

    // --------------------------------------------------------------------------
    // Uniform utility functions
    // --------------------------------------------------------------------------

    /**
     * Convenience wrapper around `this.rng.next()`
     *
     * Returns a floating point number in [0, 1).
     *
     * @return {number}
     */

  }, {
    key: 'next',
    value: function next() {
      return this._rng.next();
    }

    /**
     * Samples a uniform random floating point number, optionally specifying
     * lower and upper bounds.
     *
     * Convence wrapper around `random.uniform()`
     *
     * @param {number} [min=0] - Lower bound (float, inclusive)
     * @param {number} [max=1] - Upper bound (float, exclusive)
     * @return {number}
     */

  }, {
    key: 'float',
    value: function float(min, max) {
      return this.uniform(min, max)();
    }

    /**
     * Samples a uniform random integer, optionally specifying lower and upper
     * bounds.
     *
     * Convence wrapper around `random.uniformInt()`
     *
     * @param {number} [min=0] - Lower bound (integer, inclusive)
     * @param {number} [max=1] - Upper bound (integer, inclusive)
     * @return {number}
     */

  }, {
    key: 'int',
    value: function int(min, max) {
      return this.uniformInt(min, max)();
    }

    /**
     * Samples a uniform random integer, optionally specifying lower and upper
     * bounds.
     *
     * Convence wrapper around `random.uniformInt()`
     *
     * @alias `random.int`
     *
     * @param {number} [min=0] - Lower bound (integer, inclusive)
     * @param {number} [max=1] - Upper bound (integer, inclusive)
     * @return {number}
     */

  }, {
    key: 'integer',
    value: function integer(min, max) {
      return this.uniformInt(min, max)();
    }

    /**
     * Samples a uniform random boolean value.
     *
     * Convence wrapper around `random.uniformBoolean()`
     *
     * @alias `random.boolean`
     *
     * @return {boolean}
     */

  }, {
    key: 'bool',
    value: function bool() {
      return this.uniformBoolean()();
    }

    /**
     * Samples a uniform random boolean value.
     *
     * Convence wrapper around `random.uniformBoolean()`
     *
     * @return {boolean}
     */

  }, {
    key: 'boolean',
    value: function boolean() {
      return this.uniformBoolean()();
    }

    // --------------------------------------------------------------------------
    // Uniform distributions
    // --------------------------------------------------------------------------

    /**
     * Generates a [Continuous uniform distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)).
     *
     * @param {number} [min=0] - Lower bound (float, inclusive)
     * @param {number} [max=1] - Upper bound (float, exclusive)
     * @return {function}
     */

  }, {
    key: 'uniform',
    value: function uniform(min, max) {
      return this._memoize('uniform', _uniform3.default, min, max);
    }

    /**
     * Generates a [Discrete uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution).
     *
     * @param {number} [min=0] - Lower bound (integer, inclusive)
     * @param {number} [max=1] - Upper bound (integer, inclusive)
     * @return {function}
     */

  }, {
    key: 'uniformInt',
    value: function uniformInt(min, max) {
      return this._memoize('uniformInt', _uniformInt3.default, min, max);
    }

    /**
     * Generates a [Discrete uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution),
     * with two possible outcomes, `true` or `false.
     *
     * This method is analogous to flipping a coin.
     *
     * @return {function}
     */

  }, {
    key: 'uniformBoolean',
    value: function uniformBoolean() {
      return this._memoize('uniformBoolean', _uniformBoolean3.default);
    }

    // --------------------------------------------------------------------------
    // Normal distributions
    // --------------------------------------------------------------------------

    /**
     * Generates a [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution).
     *
     * @param {number} [mu=0] - Mean
     * @param {number} [sigma=1] - Standard deviation
     * @return {function}
     */

  }, {
    key: 'normal',
    value: function normal(mu, sigma) {
      return (0, _normal3.default)(this, mu, sigma);
    }

    /**
     * Generates a [Log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution).
     *
     * @param {number} [mu=0] - Mean of underlying normal distribution
     * @param {number} [sigma=1] - Standard deviation of underlying normal distribution
     * @return {function}
     */

  }, {
    key: 'logNormal',
    value: function logNormal(mu, sigma) {
      return (0, _logNormal3.default)(this, mu, sigma);
    }

    // --------------------------------------------------------------------------
    // Bernoulli distributions
    // --------------------------------------------------------------------------

    /**
     * Generates a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution).
     *
     * @param {number} [p=0.5] - Success probability of each trial.
     * @return {function}
     */

  }, {
    key: 'bernoulli',
    value: function bernoulli(p) {
      return (0, _bernoulli3.default)(this, p);
    }

    /**
     * Generates a [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution).
     *
     * @param {number} [n=1] - Number of trials.
     * @param {number} [p=0.5] - Success probability of each trial.
     * @return {function}
     */

  }, {
    key: 'binomial',
    value: function binomial(n, p) {
      return (0, _binomial3.default)(this, n, p);
    }

    /**
     * Generates a [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution).
     *
     * @param {number} [p=0.5] - Success probability of each trial.
     * @return {function}
     */

  }, {
    key: 'geometric',
    value: function geometric(p) {
      return (0, _geometric3.default)(this, p);
    }

    // --------------------------------------------------------------------------
    // Poisson distributions
    // --------------------------------------------------------------------------

    /**
     * Generates a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution).
     *
     * @param {number} [lambda=1] - Mean (lambda > 0)
     * @return {function}
     */

  }, {
    key: 'poisson',
    value: function poisson(lambda) {
      return (0, _poisson3.default)(this, lambda);
    }

    /**
     * Generates an [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution).
     *
     * @param {number} [lambda=1] - Inverse mean (lambda > 0)
     * @return {function}
     */

  }, {
    key: 'exponential',
    value: function exponential(lambda) {
      return (0, _exponential3.default)(this, lambda);
    }

    // --------------------------------------------------------------------------
    // Misc distributions
    // --------------------------------------------------------------------------

    /**
     * Generates an [Irwin Hall distribution](https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution).
     *
     * @param {number} [n=1] - Number of uniform samples to sum (n >= 0)
     * @return {function}
     */

  }, {
    key: 'irwinHall',
    value: function irwinHall(n) {
      return (0, _irwinHall3.default)(this, n);
    }

    /**
     * Generates a [Bates distribution](https://en.wikipedia.org/wiki/Bates_distribution).
     *
     * @param {number} [n=1] - Number of uniform samples to average (n >= 1)
     * @return {function}
     */

  }, {
    key: 'bates',
    value: function bates(n) {
      return (0, _bates3.default)(this, n);
    }

    /**
     * Generates a [Pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution).
     *
     * @param {number} [alpha=1] - Alpha
     * @return {function}
     */

  }, {
    key: 'pareto',
    value: function pareto(alpha) {
      return (0, _pareto3.default)(this, alpha);
    }

    // --------------------------------------------------------------------------
    // Internal
    // --------------------------------------------------------------------------

    /**
     * Memoizes distributions to ensure they're only created when necessary.
     *
     * Returns a thunk which that returns independent, identically distributed
     * samples from the specified distribution.
     *
     * @private
     *
     * @param {string} label - Name of distribution
     * @param {function} getter - Function which generates a new distribution
     * @param {...*} args - Distribution-specific arguments
     *
     * @return {function}
     */

  }, {
    key: '_memoize',
    value: function _memoize(label, getter) {
      for (var _len = arguments.length, args = Array(_len > 2 ? _len - 2 : 0), _key = 2; _key < _len; _key++) {
        args[_key - 2] = arguments[_key];
      }

      var key = '' + args.join(';');
      var value = this._cache[label];

      if (value === undefined || value.key !== key) {
        value = { key: key, distribution: getter.apply(undefined, [this].concat(args)) };
        this._cache[label] = value;
      }

      return value.distribution;
    }
  }, {
    key: 'rng',
    get: function get() {
      return this._rng;
    }
  }]);

  return Random;
}();

// defaults to Math.random as its RNG


exports.default = new Random();

},{"./distributions/bates":13,"./distributions/bernoulli":14,"./distributions/binomial":15,"./distributions/exponential":16,"./distributions/geometric":17,"./distributions/irwin-hall":18,"./distributions/log-normal":19,"./distributions/normal":20,"./distributions/pareto":21,"./distributions/poisson":22,"./distributions/uniform":25,"./distributions/uniform-boolean":23,"./distributions/uniform-int":24,"./rng":29,"./rng-factory":28,"ow-lite":8}],28:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _typeof = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? function (obj) { return typeof obj; } : function (obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; };

var _seedrandom = require('seedrandom');

var _seedrandom2 = _interopRequireDefault(_seedrandom);

var _rng = require('./rng');

var _rng2 = _interopRequireDefault(_rng);

var _function = require('./generators/function');

var _function2 = _interopRequireDefault(_function);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

exports.default = function () {
  for (var _len = arguments.length, args = Array(_len), _key = 0; _key < _len; _key++) {
    args[_key] = arguments[_key];
  }

  var _args$ = args[0],
      arg0 = _args$ === undefined ? 'default' : _args$,
      rest = args.slice(1);


  switch (typeof arg0 === 'undefined' ? 'undefined' : _typeof(arg0)) {
    case 'object':
      if (arg0 instanceof _rng2.default) {
        return arg0;
      }
      break;

    case 'function':
      return new _function2.default(arg0);

    case 'string':
    case 'number':
      return new _function2.default(_seedrandom2.default.apply(undefined, _toConsumableArray(rest)));
  }

  throw new Error('invalid RNG "' + arg0 + '"');
};

},{"./generators/function":26,"./rng":29,"seedrandom":31}],29:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var RNG = function () {
  function RNG() {
    _classCallCheck(this, RNG);
  }

  _createClass(RNG, [{
    key: 'next',
    value: function next() {
      throw new Error('RNG.next must be overridden');
    }
  }, {
    key: 'seed',
    value: function seed(_seed, opts) {
      throw new Error('RNG.seed must be overridden');
    }
  }, {
    key: 'clone',
    value: function clone(seed, opts) {
      throw new Error('RNG.clone must be overridden');
    }
  }, {
    key: '_seed',
    value: function _seed(seed, opts) {
      // TODO: add entropy and stuff

      if (seed === (seed | 0)) {
        return seed;
      } else {
        var strSeed = '' + seed;
        var s = 0;

        for (var k = 0; k < strSeed.length; ++k) {
          s ^= strSeed.charCodeAt(k) | 0;
        }

        return s;
      }
    }
  }, {
    key: 'name',
    get: function get() {
      throw new Error('RNG.name must be overridden');
    }
  }]);

  return RNG;
}();

exports.default = RNG;

},{}],30:[function(require,module,exports){
'use strict'

module.exports = require('./dist/random').default

},{"./dist/random":27}],31:[function(require,module,exports){
// A library of seedable RNGs implemented in Javascript.
//
// Usage:
//
// var seedrandom = require('seedrandom');
// var random = seedrandom(1); // or any seed.
// var x = random();       // 0 <= x < 1.  Every bit is random.
// var x = random.quick(); // 0 <= x < 1.  32 bits of randomness.

// alea, a 53-bit multiply-with-carry generator by Johannes Baagøe.
// Period: ~2^116
// Reported to pass all BigCrush tests.
var alea = require('./lib/alea');

// xor128, a pure xor-shift generator by George Marsaglia.
// Period: 2^128-1.
// Reported to fail: MatrixRank and LinearComp.
var xor128 = require('./lib/xor128');

// xorwow, George Marsaglia's 160-bit xor-shift combined plus weyl.
// Period: 2^192-2^32
// Reported to fail: CollisionOver, SimpPoker, and LinearComp.
var xorwow = require('./lib/xorwow');

// xorshift7, by François Panneton and Pierre L'ecuyer, takes
// a different approach: it adds robustness by allowing more shifts
// than Marsaglia's original three.  It is a 7-shift generator
// with 256 bits, that passes BigCrush with no systmatic failures.
// Period 2^256-1.
// No systematic BigCrush failures reported.
var xorshift7 = require('./lib/xorshift7');

// xor4096, by Richard Brent, is a 4096-bit xor-shift with a
// very long period that also adds a Weyl generator. It also passes
// BigCrush with no systematic failures.  Its long period may
// be useful if you have many generators and need to avoid
// collisions.
// Period: 2^4128-2^32.
// No systematic BigCrush failures reported.
var xor4096 = require('./lib/xor4096');

// Tyche-i, by Samuel Neves and Filipe Araujo, is a bit-shifting random
// number generator derived from ChaCha, a modern stream cipher.
// https://eden.dei.uc.pt/~sneves/pubs/2011-snfa2.pdf
// Period: ~2^127
// No systematic BigCrush failures reported.
var tychei = require('./lib/tychei');

// The original ARC4-based prng included in this library.
// Period: ~2^1600
var sr = require('./seedrandom');

sr.alea = alea;
sr.xor128 = xor128;
sr.xorwow = xorwow;
sr.xorshift7 = xorshift7;
sr.xor4096 = xor4096;
sr.tychei = tychei;

module.exports = sr;

},{"./lib/alea":32,"./lib/tychei":33,"./lib/xor128":34,"./lib/xor4096":35,"./lib/xorshift7":36,"./lib/xorwow":37,"./seedrandom":38}],32:[function(require,module,exports){
// A port of an algorithm by Johannes Baagøe <baagoe@baagoe.com>, 2010
// http://baagoe.com/en/RandomMusings/javascript/
// https://github.com/nquinlan/better-random-numbers-for-javascript-mirror
// Original work is under MIT license -

// Copyright (C) 2010 by Johannes Baagøe <baagoe@baagoe.org>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.



(function(global, module, define) {

function Alea(seed) {
  var me = this, mash = Mash();

  me.next = function() {
    var t = 2091639 * me.s0 + me.c * 2.3283064365386963e-10; // 2^-32
    me.s0 = me.s1;
    me.s1 = me.s2;
    return me.s2 = t - (me.c = t | 0);
  };

  // Apply the seeding algorithm from Baagoe.
  me.c = 1;
  me.s0 = mash(' ');
  me.s1 = mash(' ');
  me.s2 = mash(' ');
  me.s0 -= mash(seed);
  if (me.s0 < 0) { me.s0 += 1; }
  me.s1 -= mash(seed);
  if (me.s1 < 0) { me.s1 += 1; }
  me.s2 -= mash(seed);
  if (me.s2 < 0) { me.s2 += 1; }
  mash = null;
}

function copy(f, t) {
  t.c = f.c;
  t.s0 = f.s0;
  t.s1 = f.s1;
  t.s2 = f.s2;
  return t;
}

function impl(seed, opts) {
  var xg = new Alea(seed),
      state = opts && opts.state,
      prng = xg.next;
  prng.int32 = function() { return (xg.next() * 0x100000000) | 0; }
  prng.double = function() {
    return prng() + (prng() * 0x200000 | 0) * 1.1102230246251565e-16; // 2^-53
  };
  prng.quick = prng;
  if (state) {
    if (typeof(state) == 'object') copy(state, xg);
    prng.state = function() { return copy(xg, {}); }
  }
  return prng;
}

function Mash() {
  var n = 0xefc8249d;

  var mash = function(data) {
    data = String(data);
    for (var i = 0; i < data.length; i++) {
      n += data.charCodeAt(i);
      var h = 0.02519603282416938 * n;
      n = h >>> 0;
      h -= n;
      h *= n;
      n = h >>> 0;
      h -= n;
      n += h * 0x100000000; // 2^32
    }
    return (n >>> 0) * 2.3283064365386963e-10; // 2^-32
  };

  return mash;
}


if (module && module.exports) {
  module.exports = impl;
} else if (define && define.amd) {
  define(function() { return impl; });
} else {
  this.alea = impl;
}

})(
  this,
  (typeof module) == 'object' && module,    // present in node.js
  (typeof define) == 'function' && define   // present with an AMD loader
);



},{}],33:[function(require,module,exports){
// A Javascript implementaion of the "Tyche-i" prng algorithm by
// Samuel Neves and Filipe Araujo.
// See https://eden.dei.uc.pt/~sneves/pubs/2011-snfa2.pdf

(function(global, module, define) {

function XorGen(seed) {
  var me = this, strseed = '';

  // Set up generator function.
  me.next = function() {
    var b = me.b, c = me.c, d = me.d, a = me.a;
    b = (b << 25) ^ (b >>> 7) ^ c;
    c = (c - d) | 0;
    d = (d << 24) ^ (d >>> 8) ^ a;
    a = (a - b) | 0;
    me.b = b = (b << 20) ^ (b >>> 12) ^ c;
    me.c = c = (c - d) | 0;
    me.d = (d << 16) ^ (c >>> 16) ^ a;
    return me.a = (a - b) | 0;
  };

  /* The following is non-inverted tyche, which has better internal
   * bit diffusion, but which is about 25% slower than tyche-i in JS.
  me.next = function() {
    var a = me.a, b = me.b, c = me.c, d = me.d;
    a = (me.a + me.b | 0) >>> 0;
    d = me.d ^ a; d = d << 16 ^ d >>> 16;
    c = me.c + d | 0;
    b = me.b ^ c; b = b << 12 ^ d >>> 20;
    me.a = a = a + b | 0;
    d = d ^ a; me.d = d = d << 8 ^ d >>> 24;
    me.c = c = c + d | 0;
    b = b ^ c;
    return me.b = (b << 7 ^ b >>> 25);
  }
  */

  me.a = 0;
  me.b = 0;
  me.c = 2654435769 | 0;
  me.d = 1367130551;

  if (seed === Math.floor(seed)) {
    // Integer seed.
    me.a = (seed / 0x100000000) | 0;
    me.b = seed | 0;
  } else {
    // String seed.
    strseed += seed;
  }

  // Mix in string seed, then discard an initial batch of 64 values.
  for (var k = 0; k < strseed.length + 20; k++) {
    me.b ^= strseed.charCodeAt(k) | 0;
    me.next();
  }
}

function copy(f, t) {
  t.a = f.a;
  t.b = f.b;
  t.c = f.c;
  t.d = f.d;
  return t;
};

function impl(seed, opts) {
  var xg = new XorGen(seed),
      state = opts && opts.state,
      prng = function() { return (xg.next() >>> 0) / 0x100000000; };
  prng.double = function() {
    do {
      var top = xg.next() >>> 11,
          bot = (xg.next() >>> 0) / 0x100000000,
          result = (top + bot) / (1 << 21);
    } while (result === 0);
    return result;
  };
  prng.int32 = xg.next;
  prng.quick = prng;
  if (state) {
    if (typeof(state) == 'object') copy(state, xg);
    prng.state = function() { return copy(xg, {}); }
  }
  return prng;
}

if (module && module.exports) {
  module.exports = impl;
} else if (define && define.amd) {
  define(function() { return impl; });
} else {
  this.tychei = impl;
}

})(
  this,
  (typeof module) == 'object' && module,    // present in node.js
  (typeof define) == 'function' && define   // present with an AMD loader
);



},{}],34:[function(require,module,exports){
// A Javascript implementaion of the "xor128" prng algorithm by
// George Marsaglia.  See http://www.jstatsoft.org/v08/i14/paper

(function(global, module, define) {

function XorGen(seed) {
  var me = this, strseed = '';

  me.x = 0;
  me.y = 0;
  me.z = 0;
  me.w = 0;

  // Set up generator function.
  me.next = function() {
    var t = me.x ^ (me.x << 11);
    me.x = me.y;
    me.y = me.z;
    me.z = me.w;
    return me.w ^= (me.w >>> 19) ^ t ^ (t >>> 8);
  };

  if (seed === (seed | 0)) {
    // Integer seed.
    me.x = seed;
  } else {
    // String seed.
    strseed += seed;
  }

  // Mix in string seed, then discard an initial batch of 64 values.
  for (var k = 0; k < strseed.length + 64; k++) {
    me.x ^= strseed.charCodeAt(k) | 0;
    me.next();
  }
}

function copy(f, t) {
  t.x = f.x;
  t.y = f.y;
  t.z = f.z;
  t.w = f.w;
  return t;
}

function impl(seed, opts) {
  var xg = new XorGen(seed),
      state = opts && opts.state,
      prng = function() { return (xg.next() >>> 0) / 0x100000000; };
  prng.double = function() {
    do {
      var top = xg.next() >>> 11,
          bot = (xg.next() >>> 0) / 0x100000000,
          result = (top + bot) / (1 << 21);
    } while (result === 0);
    return result;
  };
  prng.int32 = xg.next;
  prng.quick = prng;
  if (state) {
    if (typeof(state) == 'object') copy(state, xg);
    prng.state = function() { return copy(xg, {}); }
  }
  return prng;
}

if (module && module.exports) {
  module.exports = impl;
} else if (define && define.amd) {
  define(function() { return impl; });
} else {
  this.xor128 = impl;
}

})(
  this,
  (typeof module) == 'object' && module,    // present in node.js
  (typeof define) == 'function' && define   // present with an AMD loader
);



},{}],35:[function(require,module,exports){
// A Javascript implementaion of Richard Brent's Xorgens xor4096 algorithm.
//
// This fast non-cryptographic random number generator is designed for
// use in Monte-Carlo algorithms. It combines a long-period xorshift
// generator with a Weyl generator, and it passes all common batteries
// of stasticial tests for randomness while consuming only a few nanoseconds
// for each prng generated.  For background on the generator, see Brent's
// paper: "Some long-period random number generators using shifts and xors."
// http://arxiv.org/pdf/1004.3115v1.pdf
//
// Usage:
//
// var xor4096 = require('xor4096');
// random = xor4096(1);                        // Seed with int32 or string.
// assert.equal(random(), 0.1520436450538547); // (0, 1) range, 53 bits.
// assert.equal(random.int32(), 1806534897);   // signed int32, 32 bits.
//
// For nonzero numeric keys, this impelementation provides a sequence
// identical to that by Brent's xorgens 3 implementaion in C.  This
// implementation also provides for initalizing the generator with
// string seeds, or for saving and restoring the state of the generator.
//
// On Chrome, this prng benchmarks about 2.1 times slower than
// Javascript's built-in Math.random().

(function(global, module, define) {

function XorGen(seed) {
  var me = this;

  // Set up generator function.
  me.next = function() {
    var w = me.w,
        X = me.X, i = me.i, t, v;
    // Update Weyl generator.
    me.w = w = (w + 0x61c88647) | 0;
    // Update xor generator.
    v = X[(i + 34) & 127];
    t = X[i = ((i + 1) & 127)];
    v ^= v << 13;
    t ^= t << 17;
    v ^= v >>> 15;
    t ^= t >>> 12;
    // Update Xor generator array state.
    v = X[i] = v ^ t;
    me.i = i;
    // Result is the combination.
    return (v + (w ^ (w >>> 16))) | 0;
  };

  function init(me, seed) {
    var t, v, i, j, w, X = [], limit = 128;
    if (seed === (seed | 0)) {
      // Numeric seeds initialize v, which is used to generates X.
      v = seed;
      seed = null;
    } else {
      // String seeds are mixed into v and X one character at a time.
      seed = seed + '\0';
      v = 0;
      limit = Math.max(limit, seed.length);
    }
    // Initialize circular array and weyl value.
    for (i = 0, j = -32; j < limit; ++j) {
      // Put the unicode characters into the array, and shuffle them.
      if (seed) v ^= seed.charCodeAt((j + 32) % seed.length);
      // After 32 shuffles, take v as the starting w value.
      if (j === 0) w = v;
      v ^= v << 10;
      v ^= v >>> 15;
      v ^= v << 4;
      v ^= v >>> 13;
      if (j >= 0) {
        w = (w + 0x61c88647) | 0;     // Weyl.
        t = (X[j & 127] ^= (v + w));  // Combine xor and weyl to init array.
        i = (0 == t) ? i + 1 : 0;     // Count zeroes.
      }
    }
    // We have detected all zeroes; make the key nonzero.
    if (i >= 128) {
      X[(seed && seed.length || 0) & 127] = -1;
    }
    // Run the generator 512 times to further mix the state before using it.
    // Factoring this as a function slows the main generator, so it is just
    // unrolled here.  The weyl generator is not advanced while warming up.
    i = 127;
    for (j = 4 * 128; j > 0; --j) {
      v = X[(i + 34) & 127];
      t = X[i = ((i + 1) & 127)];
      v ^= v << 13;
      t ^= t << 17;
      v ^= v >>> 15;
      t ^= t >>> 12;
      X[i] = v ^ t;
    }
    // Storing state as object members is faster than using closure variables.
    me.w = w;
    me.X = X;
    me.i = i;
  }

  init(me, seed);
}

function copy(f, t) {
  t.i = f.i;
  t.w = f.w;
  t.X = f.X.slice();
  return t;
};

function impl(seed, opts) {
  if (seed == null) seed = +(new Date);
  var xg = new XorGen(seed),
      state = opts && opts.state,
      prng = function() { return (xg.next() >>> 0) / 0x100000000; };
  prng.double = function() {
    do {
      var top = xg.next() >>> 11,
          bot = (xg.next() >>> 0) / 0x100000000,
          result = (top + bot) / (1 << 21);
    } while (result === 0);
    return result;
  };
  prng.int32 = xg.next;
  prng.quick = prng;
  if (state) {
    if (state.X) copy(state, xg);
    prng.state = function() { return copy(xg, {}); }
  }
  return prng;
}

if (module && module.exports) {
  module.exports = impl;
} else if (define && define.amd) {
  define(function() { return impl; });
} else {
  this.xor4096 = impl;
}

})(
  this,                                     // window object or global
  (typeof module) == 'object' && module,    // present in node.js
  (typeof define) == 'function' && define   // present with an AMD loader
);

},{}],36:[function(require,module,exports){
// A Javascript implementaion of the "xorshift7" algorithm by
// François Panneton and Pierre L'ecuyer:
// "On the Xorgshift Random Number Generators"
// http://saluc.engr.uconn.edu/refs/crypto/rng/panneton05onthexorshift.pdf

(function(global, module, define) {

function XorGen(seed) {
  var me = this;

  // Set up generator function.
  me.next = function() {
    // Update xor generator.
    var X = me.x, i = me.i, t, v, w;
    t = X[i]; t ^= (t >>> 7); v = t ^ (t << 24);
    t = X[(i + 1) & 7]; v ^= t ^ (t >>> 10);
    t = X[(i + 3) & 7]; v ^= t ^ (t >>> 3);
    t = X[(i + 4) & 7]; v ^= t ^ (t << 7);
    t = X[(i + 7) & 7]; t = t ^ (t << 13); v ^= t ^ (t << 9);
    X[i] = v;
    me.i = (i + 1) & 7;
    return v;
  };

  function init(me, seed) {
    var j, w, X = [];

    if (seed === (seed | 0)) {
      // Seed state array using a 32-bit integer.
      w = X[0] = seed;
    } else {
      // Seed state using a string.
      seed = '' + seed;
      for (j = 0; j < seed.length; ++j) {
        X[j & 7] = (X[j & 7] << 15) ^
            (seed.charCodeAt(j) + X[(j + 1) & 7] << 13);
      }
    }
    // Enforce an array length of 8, not all zeroes.
    while (X.length < 8) X.push(0);
    for (j = 0; j < 8 && X[j] === 0; ++j);
    if (j == 8) w = X[7] = -1; else w = X[j];

    me.x = X;
    me.i = 0;

    // Discard an initial 256 values.
    for (j = 256; j > 0; --j) {
      me.next();
    }
  }

  init(me, seed);
}

function copy(f, t) {
  t.x = f.x.slice();
  t.i = f.i;
  return t;
}

function impl(seed, opts) {
  if (seed == null) seed = +(new Date);
  var xg = new XorGen(seed),
      state = opts && opts.state,
      prng = function() { return (xg.next() >>> 0) / 0x100000000; };
  prng.double = function() {
    do {
      var top = xg.next() >>> 11,
          bot = (xg.next() >>> 0) / 0x100000000,
          result = (top + bot) / (1 << 21);
    } while (result === 0);
    return result;
  };
  prng.int32 = xg.next;
  prng.quick = prng;
  if (state) {
    if (state.x) copy(state, xg);
    prng.state = function() { return copy(xg, {}); }
  }
  return prng;
}

if (module && module.exports) {
  module.exports = impl;
} else if (define && define.amd) {
  define(function() { return impl; });
} else {
  this.xorshift7 = impl;
}

})(
  this,
  (typeof module) == 'object' && module,    // present in node.js
  (typeof define) == 'function' && define   // present with an AMD loader
);


},{}],37:[function(require,module,exports){
// A Javascript implementaion of the "xorwow" prng algorithm by
// George Marsaglia.  See http://www.jstatsoft.org/v08/i14/paper

(function(global, module, define) {

function XorGen(seed) {
  var me = this, strseed = '';

  // Set up generator function.
  me.next = function() {
    var t = (me.x ^ (me.x >>> 2));
    me.x = me.y; me.y = me.z; me.z = me.w; me.w = me.v;
    return (me.d = (me.d + 362437 | 0)) +
       (me.v = (me.v ^ (me.v << 4)) ^ (t ^ (t << 1))) | 0;
  };

  me.x = 0;
  me.y = 0;
  me.z = 0;
  me.w = 0;
  me.v = 0;

  if (seed === (seed | 0)) {
    // Integer seed.
    me.x = seed;
  } else {
    // String seed.
    strseed += seed;
  }

  // Mix in string seed, then discard an initial batch of 64 values.
  for (var k = 0; k < strseed.length + 64; k++) {
    me.x ^= strseed.charCodeAt(k) | 0;
    if (k == strseed.length) {
      me.d = me.x << 10 ^ me.x >>> 4;
    }
    me.next();
  }
}

function copy(f, t) {
  t.x = f.x;
  t.y = f.y;
  t.z = f.z;
  t.w = f.w;
  t.v = f.v;
  t.d = f.d;
  return t;
}

function impl(seed, opts) {
  var xg = new XorGen(seed),
      state = opts && opts.state,
      prng = function() { return (xg.next() >>> 0) / 0x100000000; };
  prng.double = function() {
    do {
      var top = xg.next() >>> 11,
          bot = (xg.next() >>> 0) / 0x100000000,
          result = (top + bot) / (1 << 21);
    } while (result === 0);
    return result;
  };
  prng.int32 = xg.next;
  prng.quick = prng;
  if (state) {
    if (typeof(state) == 'object') copy(state, xg);
    prng.state = function() { return copy(xg, {}); }
  }
  return prng;
}

if (module && module.exports) {
  module.exports = impl;
} else if (define && define.amd) {
  define(function() { return impl; });
} else {
  this.xorwow = impl;
}

})(
  this,
  (typeof module) == 'object' && module,    // present in node.js
  (typeof define) == 'function' && define   // present with an AMD loader
);



},{}],38:[function(require,module,exports){
/*
Copyright 2019 David Bau.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

(function (global, pool, math) {
//
// The following constants are related to IEEE 754 limits.
//

var width = 256,        // each RC4 output is 0 <= x < 256
    chunks = 6,         // at least six RC4 outputs for each double
    digits = 52,        // there are 52 significant digits in a double
    rngname = 'random', // rngname: name for Math.random and Math.seedrandom
    startdenom = math.pow(width, chunks),
    significance = math.pow(2, digits),
    overflow = significance * 2,
    mask = width - 1,
    nodecrypto;         // node.js crypto module, initialized at the bottom.

//
// seedrandom()
// This is the seedrandom function described above.
//
function seedrandom(seed, options, callback) {
  var key = [];
  options = (options == true) ? { entropy: true } : (options || {});

  // Flatten the seed string or build one from local entropy if needed.
  var shortseed = mixkey(flatten(
    options.entropy ? [seed, tostring(pool)] :
    (seed == null) ? autoseed() : seed, 3), key);

  // Use the seed to initialize an ARC4 generator.
  var arc4 = new ARC4(key);

  // This function returns a random double in [0, 1) that contains
  // randomness in every bit of the mantissa of the IEEE 754 value.
  var prng = function() {
    var n = arc4.g(chunks),             // Start with a numerator n < 2 ^ 48
        d = startdenom,                 //   and denominator d = 2 ^ 48.
        x = 0;                          //   and no 'extra last byte'.
    while (n < significance) {          // Fill up all significant digits by
      n = (n + x) * width;              //   shifting numerator and
      d *= width;                       //   denominator and generating a
      x = arc4.g(1);                    //   new least-significant-byte.
    }
    while (n >= overflow) {             // To avoid rounding up, before adding
      n /= 2;                           //   last byte, shift everything
      d /= 2;                           //   right using integer math until
      x >>>= 1;                         //   we have exactly the desired bits.
    }
    return (n + x) / d;                 // Form the number within [0, 1).
  };

  prng.int32 = function() { return arc4.g(4) | 0; }
  prng.quick = function() { return arc4.g(4) / 0x100000000; }
  prng.double = prng;

  // Mix the randomness into accumulated entropy.
  mixkey(tostring(arc4.S), pool);

  // Calling convention: what to return as a function of prng, seed, is_math.
  return (options.pass || callback ||
      function(prng, seed, is_math_call, state) {
        if (state) {
          // Load the arc4 state from the given state if it has an S array.
          if (state.S) { copy(state, arc4); }
          // Only provide the .state method if requested via options.state.
          prng.state = function() { return copy(arc4, {}); }
        }

        // If called as a method of Math (Math.seedrandom()), mutate
        // Math.random because that is how seedrandom.js has worked since v1.0.
        if (is_math_call) { math[rngname] = prng; return seed; }

        // Otherwise, it is a newer calling convention, so return the
        // prng directly.
        else return prng;
      })(
  prng,
  shortseed,
  'global' in options ? options.global : (this == math),
  options.state);
}

//
// ARC4
//
// An ARC4 implementation.  The constructor takes a key in the form of
// an array of at most (width) integers that should be 0 <= x < (width).
//
// The g(count) method returns a pseudorandom integer that concatenates
// the next (count) outputs from ARC4.  Its return value is a number x
// that is in the range 0 <= x < (width ^ count).
//
function ARC4(key) {
  var t, keylen = key.length,
      me = this, i = 0, j = me.i = me.j = 0, s = me.S = [];

  // The empty key [] is treated as [0].
  if (!keylen) { key = [keylen++]; }

  // Set up S using the standard key scheduling algorithm.
  while (i < width) {
    s[i] = i++;
  }
  for (i = 0; i < width; i++) {
    s[i] = s[j = mask & (j + key[i % keylen] + (t = s[i]))];
    s[j] = t;
  }

  // The "g" method returns the next (count) outputs as one number.
  (me.g = function(count) {
    // Using instance members instead of closure state nearly doubles speed.
    var t, r = 0,
        i = me.i, j = me.j, s = me.S;
    while (count--) {
      t = s[i = mask & (i + 1)];
      r = r * width + s[mask & ((s[i] = s[j = mask & (j + t)]) + (s[j] = t))];
    }
    me.i = i; me.j = j;
    return r;
    // For robust unpredictability, the function call below automatically
    // discards an initial batch of values.  This is called RC4-drop[256].
    // See http://google.com/search?q=rsa+fluhrer+response&btnI
  })(width);
}

//
// copy()
// Copies internal state of ARC4 to or from a plain object.
//
function copy(f, t) {
  t.i = f.i;
  t.j = f.j;
  t.S = f.S.slice();
  return t;
};

//
// flatten()
// Converts an object tree to nested arrays of strings.
//
function flatten(obj, depth) {
  var result = [], typ = (typeof obj), prop;
  if (depth && typ == 'object') {
    for (prop in obj) {
      try { result.push(flatten(obj[prop], depth - 1)); } catch (e) {}
    }
  }
  return (result.length ? result : typ == 'string' ? obj : obj + '\0');
}

//
// mixkey()
// Mixes a string seed into a key that is an array of integers, and
// returns a shortened string seed that is equivalent to the result key.
//
function mixkey(seed, key) {
  var stringseed = seed + '', smear, j = 0;
  while (j < stringseed.length) {
    key[mask & j] =
      mask & ((smear ^= key[mask & j] * 19) + stringseed.charCodeAt(j++));
  }
  return tostring(key);
}

//
// autoseed()
// Returns an object for autoseeding, using window.crypto and Node crypto
// module if available.
//
function autoseed() {
  try {
    var out;
    if (nodecrypto && (out = nodecrypto.randomBytes)) {
      // The use of 'out' to remember randomBytes makes tight minified code.
      out = out(width);
    } else {
      out = new Uint8Array(width);
      (global.crypto || global.msCrypto).getRandomValues(out);
    }
    return tostring(out);
  } catch (e) {
    var browser = global.navigator,
        plugins = browser && browser.plugins;
    return [+new Date, global, plugins, global.screen, tostring(pool)];
  }
}

//
// tostring()
// Converts an array of charcodes to a string
//
function tostring(a) {
  return String.fromCharCode.apply(0, a);
}

//
// When seedrandom.js is loaded, we immediately mix a few bits
// from the built-in RNG into the entropy pool.  Because we do
// not want to interfere with deterministic PRNG state later,
// seedrandom will not call math.random on its own again after
// initialization.
//
mixkey(math.random(), pool);

//
// Nodejs and AMD support: export the implementation as a module using
// either convention.
//
if ((typeof module) == 'object' && module.exports) {
  module.exports = seedrandom;
  // When in node.js, try using crypto package for autoseeding.
  try {
    nodecrypto = require('crypto');
  } catch (ex) {}
} else if ((typeof define) == 'function' && define.amd) {
  define(function() { return seedrandom; });
} else {
  // When included as a plain script, set up Math.seedrandom global.
  math['seed' + rngname] = seedrandom;
}


// End anonymous scope, and pass initial values.
})(
  // global: `self` in browsers (including strict mode and web workers),
  // otherwise `this` in Node and other environments
  (typeof self !== 'undefined') ? self : this,
  [],     // pool: entropy pool starts empty
  Math    // math: package containing random, pow, and seedrandom
);

},{"crypto":2}]},{},[1]);
