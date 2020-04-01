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
