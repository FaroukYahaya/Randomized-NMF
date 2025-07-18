% Test script for standard and randomized NMF

clear; clc; close all;
 
% Generate synthetic nonnegative data
m = 10000;         % number of rows
n = 10000;         % number of columns
r = 5;          % target rank

W_true = rand(m, r);
H_true = rand(r, n);
X = W_true * H_true;  % ground truth nonnegative matrix

% Initialize W and H
W0 = rand(m, r);
H0 = rand(r, n);

Tmax = 60;

fprintf('Running Standard NMF...\n');
[W1, H1, RRE1, T1] = std_NMF(X, W0, H0, Tmax);
fprintf('Running Randomized NMF...\n');
[W2, H2, RRE2, T2] = rand_NMF(X, W0, H0, r, Tmax);

figure('Position', [100, 100, 800, 600], 'Color', 'white');
color_std = [0.2, 0.4, 0.8];
color_rand = [0.8, 0.2, 0.3];
h1 = semilogy(T1, RRE1, 'Color', color_std, 'LineWidth', 2.5, 'DisplayName', 'Standard NMF');
hold on;
h2 = semilogy(T2, RRE2, 'Color', color_rand, 'LineWidth', 2.5, 'DisplayName', 'Randomized NMF');
xlabel('Time (seconds)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Relative Reconstruction Error (RRE)', 'FontSize', 14, 'FontWeight', 'bold');
title('Performance Comparison: Standard vs Randomized NMF', 'FontSize', 16, 'FontWeight', 'bold', 'Padding', 20);
legend('Location', 'best', 'FontSize', 12, 'Box', 'on', 'LineWidth', 1.2);
grid on;
set(gca, 'GridAlpha', 0.3, 'GridLineStyle', '-', 'MinorGridAlpha', 0.1);
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'Box', 'on');
xlim([0 Tmax]);
ylim([min([RRE1; RRE2]) * 0.8, max([RRE1; RRE2]) * 1.2]);
set(gca, 'Color', [0.98, 0.98, 0.98]);
set(gca, 'TickDir', 'out', 'TickLength', [0.01, 0.01]);
if length(T1) < 50
    set(h1, 'Marker', 'o', 'MarkerSize', 4, 'MarkerFaceColor', color_std, 'MarkerEdgeColor', 'white');
    set(h2, 'Marker', 's', 'MarkerSize', 4, 'MarkerFaceColor', color_rand, 'MarkerEdgeColor', 'white');
end
set(gca, 'Position', [0.12, 0.12, 0.8, 0.75]);


