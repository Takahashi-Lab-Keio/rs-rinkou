import numpy as np
import matplotlib.pyplot as plt

# 線形カルマンフィルタの定義

def kf(A, B, Bu, C, Q, R, u, y, xhat, P):
    '''
    Parameters
    ----------
    A, B, Bu, C: 対象システム
        x(k+1) = A*x(k) + B*v(k) + Bu*Δu(k)
        y(k) = C'*x(k) + w(k)

    Q, R: 雑音v, wの共分散行列．v, wは正規性白色雑音で
        E[v(k)] = E[w(k)] = 0
        E[v(k)'v(k)] = Q, E[w(k)'w(k)] = R
        であることを想定．

    u: 状態更新前時点での制御入力 u(k-1)
    y: 状態更新後時点での観測出力 y(k)

    xhat, P: 更新前の状態推定値 xhat(k-1), 誤差共分散行列 P(k-1)

    Returns
    ----------
    xhat_new: 更新後の状態推定値 xhat(k)
    P_new: 更新後の誤差共分散行列 P(k)
    G: カルマンゲイン G(k)
    '''

    # 事前推定値
    xhatm = np.dot(A, xhat) + np.dot(Bu, u)   # 状態
    Pm = np.array(np.dot(A, np.dot(P, A.T)) + np.dot(B, np.dot(Q, B.T)))  # 誤差共分散

    # カルマンゲイン行列
    S = np.dot(C, np.dot(Pm, C.T)) + R
    if np.ndim(S) == 0:
        G = np.dot(Pm, C.T) / S
    else:
        G = np.dot(np.dot(Pm, C.T), np.linalg.inv(S))

    # 事後推定値
    xhat_new = xhatm + np.dot(G, (y - np.dot(C, xhatm)))    # 状態
    if np.ndim(Pm) == 0:
        P_new = (1 - G * C) * Pm
    else:
        P_new = np.dot((np.eye(Pm.shape[0]) - np.dot(G, C)), Pm)

    return xhat_new, P_new, G

def sample():
    # 問題設定
    ## システム
    A = np.array(1.)
    b = np.array(1.)
    c = np.array(1.)

    ## 雑音
    Q = np.array(10.)
    R = np.array(10.)

    ## データ数
    N = 300

    # 観測データの生成
    ## 雑音信号の生成
    v = np.random.randn(N) * np.sqrt(Q)
    w = np.random.randn(N) * np.sqrt(R)

    ## 状態空間モデルを用いた時系列データの生成
    x = np.zeros(N)
    y = np.zeros(N)

    y[0] = np.dot(c.T, x[0].T) + w[0]
    for k in range(1, N):   # 時間更新
        x[k] = np.dot(A, x[k-1].T) + np.dot(b, v[k-1])
        y[k] = np.dot(c.T, x[k].T) + w[k]

    # カルマンフィルタによる状態推定
    ## 推定値の初期化
    xhat = np.zeros(N)

    ## 初期推定値
    P = np.array(0)
    xhat[0] = 0

    ## 推定値の時間更新
    for k in range(1, N):
        xhat[k], P, G = kf(A, b, 0, c, Q, R, 0, y[k], xhat[k-1], P)

    # 結果の表示
    fig, ax = plt.subplots()

    ax.plot(y, ":", label='measured')
    ax.plot(x, label='true')
    ax.plot(xhat, label='estimated')
    # ax.xlabel('No. of samples')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    sample()
