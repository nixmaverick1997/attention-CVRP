import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import os

df_step=[]

def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms
    
    df_step.append([avg_cost,log_likelihood.mean().item(), reinforce_loss.item(), bl_loss,step])
    step_df=pd.DataFrame.from_records(df_step,columns=['Avg_cost_cvrp{}_batch{}_baseline_{}'.format(opts.graph_size,opts.batch_size,opts.baseline),
                                                       'Log_Likelihood_cvrp{}_batch{}_baseline_{}'.format(opts.graph_size,opts.batch_size,opts.baseline),
                                                       'Actor_Loss_cvrp{}_batch{}_baseline_{}'.format(opts.graph_size,opts.batch_size,opts.baseline),
                                                       'BL_Loss_cvrp{}_batch{}_baseline_{}'.format(opts.graph_size,opts.batch_size,opts.baseline),'Step'])
    

    avg_line_plot=sns.lineplot(data=step_df,x='Step',y='Avg_cost_cvrp{}_batch{}_baseline_{}'.format(opts.graph_size,opts.batch_size,opts.baseline),color='red')
    get_fig1= avg_line_plot.get_figure()
    get_fig1.savefig(os.path.join('log_img','Avg_cost_set_cvrp{}_baseline_{}_batch{}.jpg'.format(opts.graph_size,opts.batch_size,opts.baseline)))
    get_fig1.clf()
    blloss_plot=sns.lineplot(data=step_df,x='Step',y='BL_Loss_cvrp{}_batch{}_baseline_{}'.format(opts.graph_size,opts.batch_size,opts.baseline),color='blue')
    get_fig2= blloss_plot.get_figure()
    get_fig2.savefig(os.path.join('log_img','BL_Loss_set_cvrp{}_baseline_{}_batch{}.jpg'.format(opts.graph_size,opts.batch_size,opts.baseline)))
    get_fig2.clf()
    ll_plot=sns.lineplot(data=step_df,x='Step',y='Log_Likelihood_cvrp{}_batch{}_baseline_{}'.format(opts.graph_size,opts.batch_size,opts.baseline),color='green')
    get_fig3= ll_plot.get_figure()
    get_fig3.savefig(os.path.join('log_img','Log_Likelihood_set_cvrp{}_baseline_{}_batch{}.jpg'.format(opts.graph_size,opts.batch_size,opts.baseline)))
    get_fig3.clf()
    acloss_plot=sns.lineplot(data=step_df,x='Step',y='Actor_Loss_cvrp{}_batch{}_baseline_{}'.format(opts.graph_size,opts.batch_size,opts.baseline),color='black')
    get_fig4= acloss_plot.get_figure()
    get_fig4.savefig(os.path.join('log_img','Actor_Loss_set_cvrp{}_baseline_{}_batch{}.jpg'.format(opts.graph_size,opts.batch_size,opts.baseline)))
    get_fig4.clf()
    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))


    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)
        #sns.lineplot(data=avg_cost_step,x="Average_Cost", y="Step")
        #plt.save('logdata/')

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
