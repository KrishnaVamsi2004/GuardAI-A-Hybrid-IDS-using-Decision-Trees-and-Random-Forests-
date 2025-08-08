import java.util.*;

public class BoardGame {
    
    static class Cell {
        int x, y, steps;
        Cell(int x, int y, int steps) {
            this.x = x;
            this.y = y;
            this.steps = steps;
        }
    }
    
    public static boolean isValid(int x, int y, int M, int N, int[][] grid, boolean[][] visited) {
        return x >= 0 && x < M && y >= 0 && y < N && grid[x][y] == 0 && !visited[x][y];
    }
    
    public static int minMoves(int[][] grid, int M, int N, int[] src, int[] dest, int[] moveRule) {
        int dx = moveRule[0], dy = moveRule[1];
        // Directions: forward, right (clockwise), left (counter-clockwise), backward
        int[][] directions = {
            {dx, dy},       // forward
            {dy, -dx},      // right
            {-dy, dx},      // left
            {-dx, -dy}      // backward
        };
        
        boolean[][] visited = new boolean[M][N];
        Queue<Cell> queue = new LinkedList<>();
        
        int sx = src[0], sy = src[1];
        int ex = dest[0], ey = dest[1];
        
        if (grid[sx][sy] == 1 || grid[ex][ey] == 1) return -1;
        
        visited[sx][sy] = true;
        queue.offer(new Cell(sx, sy, 0));
        
        while (!queue.isEmpty()) {
            Cell current = queue.poll();
            if (current.x == ex && current.y == ey) {
                return current.steps;
            }
            
            for (int[] dir : directions) {
                int nx = current.x + dir[0];
                int ny = current.y + dir[1];
                
                if (isValid(nx, ny, M, N, grid, visited)) {
                    visited[nx][ny] = true;
                    queue.offer(new Cell(nx, ny, current.steps + 1));
                }
            }
        }
        
        return -1;  // destination unreachable
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int M = sc.nextInt();
        int N = sc.nextInt();
        int[][] grid = new int[M][N];
        
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                grid[i][j] = sc.nextInt();
            }
        }
        
        int[] src = {sc.nextInt(), sc.nextInt()};
        int[] dest = {sc.nextInt(), sc.nextInt()};
        int[] moveRule = {sc.nextInt(), sc.nextInt()};
        
        int result = minMoves(grid, M, N, src, dest, moveRule);
        System.out.print(result);
        sc.close();
    }
}
