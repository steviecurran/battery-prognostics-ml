import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import os
from scipy.signal import savgol_filter, find_peaks

def spines(width,height,font,setup):
    if setup == 'y':
        plt.rcParams.update({'font.size': font})
        plt.figure(figsize = (width,height))
        ax = plt.gca();
    plt.setp(ax.spines.values(), linewidth=2)
    ax.tick_params(direction='in', pad = 7,length=6, width=1.5, which='major',right=True,top=True)
    ax.tick_params(direction='in', pad = 7,length=3, width=1.5, which='minor',right=True,top=True)
    return ax,font

is_resetting = False
colors = plt.cm.Set1(range(8))

def inter_plot(df,cyc_no, xpara, ypara, save_fmt):
    global is_resetting
    if is_resetting: 
        return
    
    ax,font = spines(6, 4, 12,'y')

    cyc = df[df['Cycle'] == cyc_no]
    
    if cyc.empty:
        print(f"ℹ️ No data available for Cycle {cyc_no} with current cell selections.")
        plt.close()
        return

    for i in range(1, 9):        
        tmp = cyc[cyc['Cell'] == i]
        if tmp.empty:
            continue
            
        x = tmp[xpara]
        y = tmp[ypara]
        
        #ls = 'dotted' if (4 <= i <= 6) else '-'
        if (i >=4) & (i <=6):
            ls = 'dotted'
            zorder = 2
        else:
            ls = '-'
            zorder = 1
        c_val = colors[i-1]
        
        ax.plot(x, y, color=c_val, lw=3, ls=ls, label=f"Cell {i}",zorder = zorder)   

    plt.xlabel(f"{xpara} [Cycle {cyc_no}]"); plt.ylabel(ypara)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    x_maj = ax.get_xticks()
    if len(x_maj) > 1:
        x_min = (x_maj[1] - x_maj[0]) / 5  
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_min))
    
    y_maj = ax.get_yticks()
    if len(y_maj) > 1:
        y_min = (y_maj[1] - y_maj[0]) / 5  
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_min))
    
    ax.legend(fontsize=0.8*font, loc='upper right')
    plt.tight_layout()
    
    if save_fmt != 'no':
        test = df['Test'].iloc[1]
        out = '../images/%s-%s_test=%s_cyc%d.%s' % (xpara, ypara, test,cyc_no, save_fmt)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, format=save_fmt, dpi=300)
        print('Plot written to %s' % (out))
    plt.show()
    
def summ_plot(dfc,xpara, ypara, save_fmt):
    global is_resetting
    if is_resetting: 
        return
        
    if dfc is None or dfc.empty:
        print("No data loaded yet. Please select options in the main widget first.")
     
    ax,font = spines(6, 4, 12,'y')
    
    for i in range(1, 9):        
        tmp = dfc[dfc['Cell'] == i]
        if tmp.empty:
            continue
            
        x = tmp[xpara]
        y = tmp[ypara]
    
        if (i >=4) & (i <=6):
            ls = 'dotted'
            zorder = 2
        else:
            ls = '-'
            zorder = 1
        c_val = colors[i-1]
        
        ax.plot(x, y, color=c_val, lw=3, ls=ls, label=f"Cell {i}",zorder = zorder)   

    plt.xlabel(xpara); plt.ylabel(ypara)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    x_maj = ax.get_xticks()
    if len(x_maj) > 1:
        x_min = (x_maj[1] - x_maj[0]) / 5  
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_min))
    
    y_maj = ax.get_yticks()
    if len(y_maj) > 1:
        y_min = (y_maj[1] - y_maj[0]) / 5  
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_min))
        
    ax.legend(fontsize=0.8*font, loc='upper right')
    plt.tight_layout()
    
    if save_fmt != 'no':
        test = dfc['Test'].iloc[1]
        out = '../images/%s-%s_test=%s.%s' % (xpara, ypara,test, save_fmt)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, format=save_fmt, dpi=300)
        print('Plot written to %s' % (out))
        
    plt.show()
    
def four_plot(dfc):
    font = 14
    plt.rcParams.update({'font.size': font})

    # Setup Grid Dimensions
    cols = 2
    rows = 2
    axes = ['ax' + '%s' % (x) for x in range(0, cols * rows)]
    fig, axs = plt.subplots(rows, cols, figsize=(11, 7))
    axno = dict(zip(axes, axs.flatten()))

    # Main Plotting Loop
    for d in range(0, len(axes)):
        ax = axes[d]

        plt.setp(axno[ax].spines.values(), linewidth=2)
        axno[ax].tick_params(direction='in', pad=7, length=6, width=1.5, which='major', right=True, top=True)
        axno[ax].tick_params(direction='in', pad=7, length=3, width=1.5, which='minor', right=True, top=True)

        for cell_no in sorted(dfc['Cell'].unique()):
            cell_data = dfc[dfc['Cell'] == cell_no]

            # Continuous trace style configuration
            if (4 <= cell_no <= 6):
                ls = 'dotted'
                zorder = 2
            else:
                ls = '-'
                zorder = 1

            xlabel = "Cycle Number"

            # Subplot routing
            if ax == 'ax0':
                x = cell_data['Cycle']
                y = cell_data['t_cyc']
                ylabel = r"Total cycle time, $t_{\rm cyc}$ [min]"
                #title = 'Battery Capacity Degradation'
                title = 'Operational perfomance'

            elif ax == 'ax1':
                x = cell_data['Cycle']
                # Take the absolute value so negative gradients plot cleanly upwards
                y = cell_data['Initial_V_slope'].abs()
                ylabel = r"Initial $|dV/dt|$ [V min$^{-1}$]"
                #title = 'Initial Ohmic Gradient Evolution'
                title = 'Voltage degradation'

            elif ax == 'ax2':
                x = cell_data['Cycle']
                y = cell_data['dq_dV_peak_height'] / 1e5
                ylabel = r"$dq/dV$ peak height [$\times 10^5$ C V$^{-1}$]"
                #title = 'Thermodynamic Chemical Health'
                title = 'Thermodynamic behaviour'

            elif ax == 'ax3':
                x = cell_data['Cycle']
                # FIXED: Corrected lowercase 'vti' string typo to uppercase 'VTI'
                y = cell_data['VTI'] / cell_data['t_cyc'] 
                ylabel = r"Effective average voltage [V]"
                #title = 'Grid Asset Valuation'
                title = 'Remaining discharge duration'

            axno[ax].plot(x, y, c=colors[cell_no - 1], label=f"Cell {cell_no}", ls=ls, lw=3, zorder=zorder)

        # Axis labels and layout management
        if ax == 'ax0' or ax == 'ax1':
            axno[ax].tick_params(labelbottom=False)
        else:
            axno[ax].set_xlabel(xlabel)

        axno[ax].set_ylabel(ylabel)
        axno[ax].grid(True, linestyle='--', alpha=0.5)
        axno[ax].set_title(title, fontsize=font)

        if ax == 'ax2':
            axno[ax].legend(prop={'size': int(0.8 * font)}, loc='upper right')

        x_maj = axno[ax].get_xticks()
        if len(x_maj) > 1:
            x_min = (x_maj[1] - x_maj[0]) / 5  
            axno[ax].xaxis.set_minor_locator(ticker.MultipleLocator(x_min))

        y_maj = axno[ax].get_yticks()
        if len(y_maj) > 1:
            y_min = (y_maj[1] - y_maj[0]) / 5  
            axno[ax].yaxis.set_minor_locator(ticker.MultipleLocator(y_min))

    fig.align_ylabels([axno['ax0'], axno['ax2']])
    fig.align_ylabels([axno['ax1'], axno['ax3']])

    plt.tight_layout()

    save_fmt = 'no' # CHANGE ACCORDINGLY
    if save_fmt != "no":
        out = "../images/4_panel_summary_test=%s.%s" %(test,save_fmt)
        plt.savefig(out, format=save_fmt, dpi=300, bbox_inches='tight')
        print('💾 Dashboard successfully exported to %s' % (out))
    plt.show()


target_cycles = [100, 4000,4800]
colours = ['lime','orange','red']

def peak_plot(df,cell,save_fmt):
    global is_resetting
    if is_resetting: 
        return
    
    ax,font = spines(8,6,14,'y')
        
    for cycle, colour in zip(target_cycles, colours):
        raw_cycle = df[(df["Cell"] == cell) & (df["Cycle"] == cycle)]

        n_points = len(raw_cycle)

        # Base target window is 15, but cannot exceed n_points
        plot_window = 15
        if n_points <= plot_window:
            plot_window = n_points if n_points % 2 != 0 else n_points - 1

        # Ensure window is valid for polyorder=2 (must be >= 5 to be reliable)
        if plot_window >= 5:
            v_smooth = savgol_filter(
                raw_cycle["Voltage_V"], plot_window, polyorder=2, mode="interp")
        else:
            v_smooth = raw_cycle["Voltage_V"].values

        # Proceed safely with differentials
        dV = np.diff(v_smooth)
        dq = np.diff(raw_cycle["Charge_mAh"])
        dq_dV = np.where(dV != 0, dq / dV, 0)

        #Plot the full continuous transformation curve safely
        ax.plot(raw_cycle["Voltage_V"].iloc[:-1],np.abs(dq_dV),
                label=f"Cycle {cycle}",c=colour,lw=1)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Cell %d Electrochemical Fingerprint Evolution" %(cell),
                 fontsize = font)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Incremental Capacity |dq/dV| [mAh V$^{-1}$]")
    ax.legend(fontsize = 0.8*font,frameon=True)
    plt.tight_layout()
    if save_fmt != 'no':
        out = '../images/EC_fingerprint_cell=%d_test=%s.%s' % (cell,test, save_fmt)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, format=save_fmt, dpi=300)
        print('Plot written to %s' % (out))
        
    plt.show()
    
def ML_plot(eval_df,y_eval,y_pred,cell,cohort_type,save_fmt):
    global is_resetting
    if is_resetting: 
        return

    ax,font = spines(10,5,15,'y')

    ax.plot(eval_df['Cycle'], y_eval, label='Actual Cycle Duration ($t_{cyc}$)', color='black', lw=2)
    ax.plot(eval_df['Cycle'], y_pred, label='Random Forest Prediction', color='crimson', ls='--', lw=2)
        
    ax.set_title(f"Cell {cell} RUL Track Verification ({cohort_type})", fontsize = font)
    ax.set_xlabel("Cycle Number"); ax.set_ylabel("Cycle Time, $t_{cyc}$ [min]")
    x_maj = ax.get_xticks()
    if len(x_maj) > 1:
        x_min = (x_maj[1] - x_maj[0]) / 5  
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_min))
    
    y_maj = ax.get_yticks()
    if len(y_maj) > 1:
        y_min = (y_maj[1] - y_maj[0]) / 5  
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_min))
        
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize = 0.8*font,loc='lower left')
    plt.tight_layout()
    if save_fmt != 'no':
        out = '../images/ML_cell=%d.%s' % (cell,save_fmt)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, format=save_fmt, dpi=300)
        print('Plot written to %s' % (out))
    plt.show()

##############################################################################
    
def anomaly_plot(tracking_df, target_cell, cohort_type, save_fmt):

    ax,font = spines(10,5,15,'y')
    
    cycles = tracking_df['Cycle'].values
    scores = tracking_df['Anomaly_Score'].values
    
    ax.plot(cycles, scores, color='b', lw=2.5, label='Battery Safety Score Trace', zorder=1)

    anomalous_data = tracking_df[tracking_df['Status'] == '🚨 ANOMALY RISK']
    
    if not anomalous_data.empty:
        ax.scatter(anomalous_data['Cycle'].values, anomalous_data['Anomaly_Score'].values, color='r', 
            s=60,marker='o',edgecolors='black',linewidths=0.7,label='Flagged Anomaly Trigger Zone',
            zorder=2
        )

    ax.axhline(0, color='red', ls=':', lw=3, alpha=0.6, label='Anomaly Threshold Limit (0.0)')

    ax.set_title(f"Cell {target_cell} Anomaly Detection ({cohort_type})", fontsize=font, pad=15)
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Anomaly Safety Score\n(Negative = High Risk Zone)")
    ax.grid(True, linestyle='--', alpha=0.4)

    x_maj = ax.get_xticks()
    if len(x_maj) > 1:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator((x_maj[1] - x_maj[0]) / 5))
        
    y_maj = ax.get_yticks()
    if len(y_maj) > 1:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator((y_maj[1] - y_maj[0]) / 5))

    ax.legend(fontsize = 0.8*font, loc='lower left', frameon=True)
    plt.tight_layout()

    if save_fmt != 'no':
        out_path = f"../images/cell{target_cell}_anomaly_trajectory.{save_fmt}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, format=save_fmt, dpi=300, bbox_inches='tight')
        print(f"💾 Anomaly trajectory plot written successfully to {out_path}")

    plt.show()

##############################################################################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

def ML_plot(data, target, regressor):

    #print(data)
    font = 16
    plt.rcParams.update({'font.size': font})

    colors = plt.cm.Set1(range(8))

    # Setup Grid Dimensions (2 rows, 4 columns)
    cols = 4; rows = 2
    axes = ['ax' + '%s' % (x) for x in range(0, cols * rows)]
    fig, axs = plt.subplots(rows, cols, figsize=(14, 7))
    axno = dict(zip(axes, axs.flatten()))
    
    data = data[data["Model"] == regressor]
    cells = sorted(data["Cell"].unique())

    # Establish axis boundary limits and dynamic string labels outside the loop
    if target == 't_cyc':
        xmin = 30; xmax = 60
        xlabel_text = r'Actual cycle time, $t_{\rm cyc}$ [mins]'
        ylabel_text = r'Predicted, $t_{\rm pred}$ [mins]'

    elif target == 'SOH':
        xmin = 0; xmax = 1
        xlabel_text = r'Actual SOH'
        ylabel_text = r'Predicted SOH'
    else:
        xmin = 0; xmax = 8500
        xlabel_text = r'Actual RUL [drive cycles]'
        ylabel_text = r'Predicted RUL [drive cycles]'

    for d in range(len(cells)):
        ax = axes[d]
        cell_no = cells[d]
        cell_data = data[data["Cell"] == cell_no]

        plt.setp(axno[ax].spines.values(), linewidth=2)
        axno[ax].tick_params(direction='in', pad=7, length=6, width=1.5, which='major', right=True, top=True)
        axno[ax].tick_params(direction='in', pad=7, length=3, width=1.5, which='minor', right=True, top=True)
        
        y_test = cell_data['Actual']
        y_pred = cell_data['Predicted']

        rms = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        
        r2 = r2_score(y_test, y_pred)
        r_val = np.corrcoef(y_test, y_pred)[0, 1] if np.std(y_pred) > 0 and np.std(y_test) > 0 else 0.0
        p = np.polyfit(y_test, y_pred, 1)
        m = p[0]; c = p[1]

        label=r"Test Cell %d" %(cell_no) + "\n"+ "RMS = %1.3g" %(rms) + "\n"+ r"MedAE = %1.3g" %(medae) \
            + "\n"+ r"$r_{\rm Pearson} = %1.3f$" %(r_val) + "\n"+  "$R^2 = %1.3f$" %(r2)  + "\n" \
            + "$m  = %1.3g$" %(m) + "\n" + "$c = %1.3g$" %(c)
        
        
        axno[ax].scatter(y_test, y_pred, color=colors[d], marker='o', s=25, label=label,zorder=1)
              
        axno[ax].legend(fontsize=0.8*font, markerscale=0, handlelength=0, handletextpad=0.2, loc='best')
        axno[ax].axline((0, 0), slope=1, linestyle='--', color='silver', lw=2,zorder=2) # y = x Identity Line
        axno[ax].axline((0, 0), slope=1, linestyle='dotted', color='k', lw=2,zorder=2) # more visible
        
        # CLEAN EXTERIOR LABELS ONLY: Hide interior tick labels to avoid cluster clipping
        row_idx = d // cols
        col_idx = d % cols
        
        if row_idx == 0:  # Top row subplots don't need bottom labels
            axno[ax].tick_params(labelbottom=False)
        if col_idx > 0:   # Right-side subplots don't need vertical labels
            axno[ax].tick_params(labelleft=False)

        axno[ax].set_xlim([xmin, xmax])
        axno[ax].set_ylim([xmin, xmax])

        # Precise minor tick subdivisions
        x_maj = axno[ax].get_xticks()
        if len(x_maj) > 1:
            x_min = (x_maj[1] - x_maj[0]) / 5  
            axno[ax].xaxis.set_minor_locator(ticker.MultipleLocator(x_min))
    
        y_maj = axno[ax].get_yticks()
        if len(y_maj) > 1:
            y_min = (y_maj[1] - y_maj[0]) / 5  
            axno[ax].yaxis.set_minor_locator(ticker.MultipleLocator(y_min))
    
    if "t_end" in data.columns and len(data) > 0:
        extraction_win = round(data["t_end"].iloc[0], 0)
        window_title_addon = f": first {extraction_win:.0f} mins"
    else:
        window_title_addon = ""

    fig.suptitle(f"{regressor} Cross-Validation Predictions vs. Measurement ({target}){window_title_addon}", 
                 fontsize=1.3*font, y=0.98)
    
    
    fig.supxlabel(xlabel_text, fontsize=1.3*font, y=0.02)
    fig.supylabel(ylabel_text, fontsize=1.3*font, x=0.01)

    plt.tight_layout(pad=0.5)
    fig.align_ylabels(axs)
    
    plt.show()


def meta_plot(data):
    if data is None or data.empty:
        print("Ingested dataset frame context is empty.")
        return

    df_plot = data.copy()
    
    # ─── THE SMART METRIC ALIGNMENT SHIELD ──────────────────────────────────
    # Check if the CSV columns already represent an aggregated summary.
    # If they do, we map the columns cleanly to  target plotting keys.
    if "RMSE_frac_mean" in df_plot.columns:
        print("Ingested data is already aggregated. Aligning structural summary tokens...")
        
        # Build out the exact un-grouped index layout averages expect
        averages = pd.DataFrame(index=df_plot["Window_Minutes"].unique())
        averages.index.name = "Window_Minutes"
        
        # Temporarily pivot the summary rows directlyonto  target keys
        for win in averages.index:
            slice_win = df_plot[df_plot["Window_Minutes"] == win].iloc[0]
            averages.loc[win, "RMSE_mean"] = slice_win["RMSE_frac_mean"]
            averages.loc[win, "RMSE_std"]  = slice_win.get("RMSE_std", 0.0) # Fallback if missing
            averages.loc[win, "MedAE_mean"] = slice_win.get("MAE_frac_mean", slice_win["MAE_mean"]) # Safe median proxy
            averages.loc[win, "MAE_std"]   = slice_win.get("MAE_std", 0.0)
            averages.loc[win, "PC_mean"]   = slice_win.get("R2_mean", 0.0)**0.5 # Proxy for correlation mean
            averages.loc[win, "PC_std"]    = slice_win.get("R2_std", 0.0)
            averages.loc[win, "R2_mean"]   = slice_win["R2_mean"]
            averages.loc[win, "R2_std"]    = slice_win.get("R2_std", 0.0)
            averages.loc[win, "m_mean"]    = slice_win.get("gradient_mean", 1.0)
            averages.loc[win, "m_std"]     = slice_win.get("gradient_std", 0.0)
            averages.loc[win, "c_mean"]    = slice_win.get("intercept_mean", 0.0)
            averages.loc[win, "c_std"]     = slice_win.get("intercept_std", 0.0)
            
    else:
        print(" Running Live Meta Groupby Aggregation...")
        
        r_col = "Pearson_coeff" if "Pearson_coeff" in df_plot.columns else ("r" if "r" in df_plot.columns else "R2")
        medae_col = "MedAE_frac" if "MedAE_frac" in df_plot.columns else "MAE_frac"
        
        averages = df_plot.groupby('Window_Minutes').agg(
            RMSE_mean=("RMSE_frac", "mean"), RMSE_std=("RMSE_frac", "std"),
            MedAE_mean=(medae_col, "mean"), MAE_std=(medae_col, "std"),
            PC_mean=(r_col, "mean"), PC_std=(r_col, "std"),
            R2_mean=("R2", "mean"), R2_std=("R2", "std"),
            m_mean=("m", "mean"), m_std=("m", "std"),
            c_mean=("c", "mean"), c_std=("c", "std")
        )
    # ──────────────────────────────────────────────────────────────────────────

    averages['No_trials'] = df_plot['Trial_ID'].iloc[-1] if 'Trial_ID' in df_plot.columns else 1
    averages['Model'] = df_plot['Model'].iloc[-1] if 'Model' in df_plot.columns else 'Model'
    averages['Target'] = df_plot['Target'].iloc[-1] if 'Target' in df_plot.columns else 'Target'

    labels = [r'${\rm RMS}/\overline{y_{\rm test}}$', r'${\rm MedAE}/\overline{y_{\rm test}}$', 
              r'$r_{\rm Pearson}$', '$R^2$', 'Gradient, $m$', 'Intercept, $c$']
    
    font = 15
    plt.rcParams.update({'font.size': font})

    cols = 3; rows = 2
    axes = ['ax' + '%s' % (x) for x in range(0, cols * rows)]
    fig, axs = plt.subplots(rows, cols, figsize=(12, 6))
    axno = dict(zip(axes, axs.flatten()))

   
    colors = plt.cm.Set1(range(8))
    metrics = [c for c in averages.columns if c not in ['No_trials', 'Model', 'Target']]
    x = averages.index
    
    for d in range(cols * rows):
        ax = axes[d]
        para = metrics[2 * d]
        err = metrics[(2 * d) + 1]
        y_val = averages[para]
        
        
        trials_count = float(averages['No_trials'].iloc[0])
        e = averages[err] / (trials_count ** 0.5) if trials_count > 1 else averages[err]
        
        plt.setp(axno[ax].spines.values(), linewidth=2)
        axno[ax].tick_params(direction='in', pad=7, length=6, width=1.5, which='major', right=True, top=True)
        axno[ax].tick_params(direction='in', pad=7, length=3, width=1.5, which='minor', right=True, top=True)
        
        axno[ax].errorbar(x, y_val, yerr=e, linestyle="", color="k", lw=2, capsize=3, zorder=1) 
        axno[ax].scatter(x, y_val, color=colors[d], ec='k', marker='o', s=65, zorder=2, label="%s" % (labels[d]))
        axno[ax].legend(fontsize=0.75 * font, markerscale=0, handlelength=0, handletextpad=0.2, loc='best')
         

        #### LOG SCALE ####
        def update_ticks(z, pos):
            return "%1.0f" %(z)
            
            '''
            if z >= -3 and z <= 3:
                return "%1.0f" %(10**z)
            else:
                return "$10^{%d}$" %(z)
            '''
        axno[ax].set_xscale('log')
        axno[ax].xaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
        #####################
        
        y_maj = axno[ax].get_yticks()
        if len(y_maj) > 1:
            y_min = (y_maj[1] - y_maj[0]) / 5  
            axno[ax].yaxis.set_minor_locator(ticker.MultipleLocator(y_min))

    
        if d < cols:
            axno[ax].tick_params(labelbottom=False)
    

    fig.suptitle('%s metrics over %d trials for %s model' %(data['Target'].iloc[-1],data['Trial_ID'].iloc[-1],
                                                            data['Model'].iloc[-1]), fontsize=1.1*font, y=1)
    fig.supxlabel('Observation window [mins]', fontsize=1.1*font)
    fig.supylabel('Value', fontsize=1.1*font,x=0.01)
    
    #plt.tight_layout(rect=[0.03, 0.04, 1, 0.99])
    plt.tight_layout(pad=0.5)
    
    fig.align_ylabels(axs)

    model_name = 'RF' if averages['Model'].iloc[0] == 'Random Forest' else 'to_fix'
    out_dir = "images"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/meta_metrics_target={averages['Target'].iloc[0]}_model={model_name}_trials={int(averages['No_trials'].iloc[0])}.png"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"💾 Plot successfully written to {out_path}")
    plt.show()
