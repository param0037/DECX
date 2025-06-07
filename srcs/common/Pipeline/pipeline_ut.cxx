#include "pipeline.h"
#include "nodes/task_node.h"
#include <basic.h>
#include <thread_management/thread_pool.h>

#define MODULE_TAG "Pipeline"


struct node_proc_params_t
{
    char* msg;
    int32_t predicator;
};


int32_t node_func1(const void* in, void* swap, void* out)
{
    auto* p_in = (node_proc_params_t*)swap;
    DECX_LOG_NOTICE(p_in->msg);
    p_in->predicator++;
    snprintf(p_in->msg, 18, "Hello from node 2");
    return 0;
}

int32_t node_func2(const void* in, void* swap, void* out)
{
    auto* p_in = (node_proc_params_t*)swap;
    DECX_LOG_NOTICE(p_in->msg);
    p_in->predicator++;
    // p_in->predicator++;
    return 0;
}


int32_t node_func_branch1(const void* in, void* swap, void* out)
{
    DECX_LOG_NOTICE("Hello from node branch 1");
    return 0;
}


int32_t node_func_branch2(const void* in, void* swap, void* out)
{
    DECX_LOG_NOTICE("Hello from node branch 2");
    return 0;
}

int32_t node_func3(const void* in, void* swap, void* out)
{
    DECX_LOG_NOTICE("Hello from node 3");
    return 0;
}


int32_t predicator_func1(const void* in, int32_t* p_slot_idx)
{
    auto* p_in = (node_proc_params_t*)in;
    if (p_in->predicator % 2){
        DECX_LOG_NOTICE("Choose slot 1");
        *p_slot_idx = 1;
    }
    else{
        DECX_LOG_NOTICE("Choose slot 2");
        *p_slot_idx = 0;
    }
    return 0;
}


void test_thread_func()
{
    DECX_LOG_INFO("test_thread_funcis called");
}


_DECX_API_ void pipeline_ut()
{
    decx::utils::Pipeline pipeline;

    char msg[18] = "Hello from node 1";

    node_proc_params_t node_param;
    node_param.msg = (char*)msg;
    node_param.predicator = 0;
    decx::utils::TaskNode* node1 = new decx::utils::TaskNode("node1");
    node1->NodeTaskRegister(node_func1);
    node1->SetData(decx::utils::TaskNode_WorkingData_Type_e::TaskNode_Data_Swap, &node_param, sizeof(node_proc_params_t));

    decx::utils::TaskNode* node2 = new decx::utils::TaskNode("node2");
    node2->NodeTaskRegister(node_func2);
    node2->SetData(decx::utils::TaskNode_WorkingData_Type_e::TaskNode_Data_Swap, &node_param, sizeof(node_proc_params_t));

    decx::utils::BranchSplit* predicator1 = new decx::utils::BranchSplit("predicator1");
    predicator1->SetPredicatedData(&node_param, sizeof(node_proc_params_t));
    predicator1->PredicatorRegister(predicator_func1);

    decx::utils::ConcurrentSplit* conc_split = new decx::utils::ConcurrentSplit("conc_split");
    decx::utils::Synchronize* sync = new decx::utils::Synchronize("sync");
    decx::utils::Synchronize* sync1 = new decx::utils::Synchronize("sync1");

    decx::utils::TaskNode* node_branch1 = new decx::utils::TaskNode("node_branch_1");
    node_branch1->NodeTaskRegister(node_func_branch1);

    decx::utils::TaskNode* node_branch2 = new decx::utils::TaskNode("node_branch_2");
    node_branch2->NodeTaskRegister(node_func_branch2);

    decx::utils::TaskNode* node3 = new decx::utils::TaskNode("node3");
    node3->NodeTaskRegister(node_func3);

    // pipeline.LinkNodes({node1, node2, predicator1});
    // pipeline.LinkBranch(predicator1, {node_branch1});
    // pipeline.LinkBranch(predicator1, {node_branch2});

    pipeline.LinkNodes({node1, node2, conc_split, sync, node3, sync1});
    pipeline.LinkStream(conc_split, {node_branch1}, sync);
    pipeline.LinkStream(conc_split, {node_branch2}, sync1);

    // std::future<void> fut = decx::cpu::RegisterTaskAppened(test_thread_func);
    // std::future<void> fut_copy;
    // memcpy(&fut_copy, &fut, sizeof(std::future<void>));
    // fut_copy.get();

    pipeline.Run();
}