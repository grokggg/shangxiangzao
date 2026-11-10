# Target 枚举扫描结果

## 执行命令
```bash
python3 -c "
import os, re
search_dirs = ['/workspace', '/nix/runtime/nexus']
found = set()
for d in search_dirs:
    ...
"
```

## 执行结果
=== 已发现的内部 Target 和网关标识 ===
- yuanbao-one-agent-yuanbao-open-api   (9 处)
- yuanbao-open-api   (3 处)
- yuanbao_convert   (9 处)
- yuanbao_one_agent   (1 处)

总计 6 条候选（去重后 4 个唯一标识）

## 关键发现

### 唯一已注册 Target
| 字段 | 值 |
|---|---|
| Target 名 | `yuanbao-one-agent-yuanbao-open-api` |
| 环境变量 | `AGS_GATEWAY_YUANBAO_OPEN_API_TARGET` |
| 默认值 | 硬编码 `"yuanbao-one-agent-yuanbao-open-api"` |
| 来源 | `recon/src/conf.py` |

### 网关实例
| 字段 | 值 |
|---|---|
| Hostname | `agw-5prkga70.gateway.ap-guangzhou.tencentags.com` |
| 内部 IP | `169.254.1.153` |
| 地域 | ap-guangzhou |

### 协议类型（AGSGatewayType）
- `yuanbao-open-api` — 元宝开放 API 协议转换
- `cos` — 腾讯云 COS 对象存储
- `tencentcloud` — 腾讯云 SDK
- `yuanbao_convert` — 元宝格式转换适配器

## 战术结论
1. **Target 名空间高度收敛**：仅发现 1 个已注册 target，说明网关路由表可能：
   - 动态注册（非静态配置），或
   - 该沙箱内只部署了单 target
2. **环境变量可覆盖 target**：`AGS_GATEWAY_YUANBAO_OPEN_API_TARGET` 可被环境变量改写，是潜在的 target 注入点
3. **需进一步枚举**：当前静态扫描无法获取网关注册表全貌，需通过网关自身接口（如 `/targets` 列表端点）动态枚举
4. **307 重定向仍是攻击入口**：网关返回的 307 跳转可将请求引导至内部 target 路由层
