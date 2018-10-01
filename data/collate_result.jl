using JSON
using HitchhikingDrones

sdmc_fn = ARGS[1]
mpc_fn = ARGS[2]
num_eps = parse(Int64, ARGS[3])
out_prefix = ARGS[4]

sdmc_res_dict = Dict()
mpc_res_dict = Dict()

open(sdmc_fn,"r") do f
    global sdmc_res_dict
    sdmc_res_dict = JSON.parse(f)
end

open(mpc_fn,"r") do f
    global mpc_res_dict
    mpc_res_dict = JSON.parse(f)
end

sdmc_avg_dict = Dict()
mpc_avg_dict = Dict()



num_valid_eps = 0


for i = 1:num_eps
    sdmc_res = sdmc_res_dict[string(i)]
    mpc_res = mpc_res_dict[string(i)]

    if sdmc_res["success"] == false || mpc_res["success"] == false
        continue
    end

    num_valid_eps += 1
end

sdmc_vals = zeros(num_valid_eps,8)
mpc_vals = zeros(num_valid_eps,6)

idx = 1

for i = 1:num_eps
    sdmc_res = sdmc_res_dict[string(i)]
    mpc_res = mpc_res_dict[string(i)]

    if sdmc_res["success"] == false || mpc_res["success"] == false
        continue
    end

    sdmc_vals[idx,1] = -sdmc_res["reward"]
    sdmc_vals[idx,2] = sdmc_res["distance"]
    sdmc_vals[idx,3] = sdmc_res["sl_dist"]
    sdmc_vals[idx,4] = sdmc_res["time"]
    sdmc_vals[idx,5] = sdmc_res["attempted_hops"]
    sdmc_vals[idx,6] = sdmc_res["successful_hops"]
    sdmc_vals[idx,7], sdmc_vals[idx,8] = st_line_reward_time(sdmc_res["sl_dist"])

    mpc_vals[idx,1] = -mpc_res["reward"]
    mpc_vals[idx,2] = mpc_res["distance"]
    mpc_vals[idx,3] = mpc_res["sl_dist"]
    mpc_vals[idx,4] = mpc_res["time"]
    mpc_vals[idx,5] = mpc_res["attempted_hops"]
    mpc_vals[idx,6] = mpc_res["successful_hops"]

    idx += 1
end

sdmc_avg_dict["avg_cost"] = (mean(sdmc_vals[:,1]), std(sdmc_vals[:,1])/sqrt(num_valid_eps))
sdmc_avg_dict["avg_dist"] = (mean(sdmc_vals[:,2]), std(sdmc_vals[:,2])/sqrt(num_valid_eps))
sdmc_avg_dict["avg_sl_dist"] = (mean(sdmc_vals[:,3]), std(sdmc_vals[:,3])/sqrt(num_valid_eps))
sdmc_avg_dict["avg_time"] = (mean(sdmc_vals[:,4]), std(sdmc_vals[:,4])/sqrt(num_valid_eps))
sdmc_avg_dict["avg_att_hops"] = (mean(sdmc_vals[:,5]), std(sdmc_vals[:,5])/sqrt(num_valid_eps))
sdmc_avg_dict["avg_succ_hops"] = (mean(sdmc_vals[:,6]), std(sdmc_vals[:,6])/sqrt(num_valid_eps))
sdmc_avg_dict["avg_st_line_cost"] = (mean(sdmc_vals[:,7]), std(sdmc_vals[:,7])/sqrt(num_valid_eps))
sdmc_avg_dict["avg_st_line_time"] = (mean(sdmc_vals[:,8]), std(sdmc_vals[:,8])/sqrt(num_valid_eps))

mpc_avg_dict["avg_cost"] = (mean(mpc_vals[:,1]), std(mpc_vals[:,1])/sqrt(num_valid_eps))
mpc_avg_dict["avg_dist"] = (mean(mpc_vals[:,2]), std(mpc_vals[:,2])/sqrt(num_valid_eps))
mpc_avg_dict["avg_sl_dist"] = (mean(mpc_vals[:,3]), std(mpc_vals[:,3])/sqrt(num_valid_eps))
mpc_avg_dict["avg_time"] = (mean(mpc_vals[:,4]), std(mpc_vals[:,4])/sqrt(num_valid_eps))
mpc_avg_dict["avg_att_hops"] = (mean(mpc_vals[:,5]), std(mpc_vals[:,5])/sqrt(num_valid_eps))
mpc_avg_dict["avg_succ_hops"] = (mean(mpc_vals[:,6]), std(mpc_vals[:,6])/sqrt(num_valid_eps))

sdmc_out_fn = string(out_prefix,"-sdmc-results.json")
open(sdmc_out_fn,"w") do f
    JSON.print(f,sdmc_avg_dict,2)
end

mpc_out_fn = string(out_prefix,"-mpc-results.json")
open(mpc_out_fn,"w") do f
    JSON.print(f,mpc_avg_dict,2)
end