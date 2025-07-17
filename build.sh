mkdir -p build
cd build

# 配置项目并启用编译命令导出 compile_commands.json, 保证 vscode 可以跳转
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

cmake ../
make -j`nproc`
cd ../

# 在项目根目录创建符号链接。检查目标文件是否存在，如果不存在才创建符号链接
if [ ! -e compile_commands.json ]; then
    echo "Creating symbolic link to compile_commands.json..."
    ln -s build/compile_commands.json .
    echo "Symbolic link created successfully."
else
    echo "compile_commands.json already exists, skipping symbolic link creation."
fi
