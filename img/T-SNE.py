
syn_feature_all, syn_label_all = generate_syn_feature_all(netG, data.allclasses, data.attribute, 50, netV=netV,
                                                              netF=netF)
# CUB
selected_seen_classes = [11, 69, 71, 1, 54, 89, 140, 100, 174, 117, 46, 95, 131, 129, 53, 25, 118, 10, 102, 163]
    
selected_unseen_classes = [5, 30, 142, 80, 9, 49, 13, 20, 7, 90, 87, 26, 161, 143, 94, 43, 191, 200, 77, 33]

    
batch_res_cpu = syn_feature_all.numpy()
batch_label_cpu = syn_label_all.numpy()
    
seen_mask = np.isin(batch_label_cpu, selected_seen_classes)
unseen_mask = np.isin(batch_label_cpu, selected_unseen_classes)
mask = seen_mask | unseen_mask
filtered_res = batch_res_cpu[mask]
filtered_labels = batch_label_cpu[mask]
    

tsne = TSNE(n_components=2, perplexity=10, random_state=42)
res_tsne = tsne.fit_transform(filtered_res)
    
plt.figure(figsize=(10, 8))
plt.style.use('seaborn')  # 使用seaborn主题
plt.rcParams['font.family'] = 'Times New Roman'  # 统一字体
plt.rcParams['axes.labelsize'] = 12
    
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.5)
    

seen_colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#FF00FF", "#00FFFF", "#800000", "#008000",
    "#000080", "#808000", "#800080", "#008080",
    "#C0C0C0", "#808080", "#FFA500", "#00FF7F",
    "#7FFFD4", "#FF69B4", "#BA55D3", "#F0E68C"
]
    
unseen_colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#FF00FF", "#00FFFF", "#800000", "#008000",
    "#000080", "#808000", "#800080", "#008080",
    "#C0C0C0", "#808080", "#FFA500", "#00FF7F",
    "#7FFFD4", "#FF69B4", "#BA55D3", "#F0E68C"
]
    
# for i, label in enumerate(selected_seen_classes):
#     mask = filtered_labels == label
#     plt.scatter(
#         res_tsne[mask, 0], res_tsne[mask, 1],
#         c=[seen_colors[i % len(seen_colors)]],
#         alpha=1,
#         s=100,
#         marker='o',  # 圆形标记
#         zorder=3
#     )
    
for i, label in enumerate(selected_unseen_classes):
    mask = filtered_labels == label
    plt.scatter(
        res_tsne[mask, 0], res_tsne[mask, 1],
        c=[unseen_colors[i % len(unseen_colors)]],
        alpha=1,
        s=100,
        marker='o',  # 圆形标记
        zorder=3
  )
    
plt.gca().set_facecolor('#ffffff')
plt.grid(color='white', linestyle='--', linewidth=0.8, alpha=0.8)
    
plt.xticks([])
plt.yticks([])
    
plt.tight_layout()
plt.savefig('O_CUBU.png', dpi=300, bbox_inches='tight')
plt.savefig('O_CUBU.pdf', bbox_inches='tight')
plt.close()
