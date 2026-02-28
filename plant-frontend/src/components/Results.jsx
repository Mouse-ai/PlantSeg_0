// src/components/Results.jsx
const Results = ({ predictions }) => {
  if (!predictions || predictions.length === 0) {
    return <p className="no-results">Ничего не обнаружено.</p>;
  }

  // Группируем данные по классам
  const groups = {
    leaf: { confSum: 0, count: 0, area: 0, length: null },
    stem: { confSum: 0, count: 0, area: 0, length: 0 },
    root: { confSum: 0, count: 0, area: 0, length: 0 }
  };

  predictions.forEach(p => {
    const cls = p.class;
    if (cls in groups) {
      groups[cls].confSum += p.confidence;
      groups[cls].count += 1;
      groups[cls].area += p.area_cm2 || 0;
      if (p.length_cm != null && !isNaN(p.length_cm)) {
        groups[cls].length += p.length_cm;
      }
    }
  });

  // Средняя уверенность в процентах
  const avgConf = (group) => {
    if (group.count === 0) return '—';
    return (group.confSum / group.count * 100).toFixed(1) + '%';
  };

  const formatArea = (v) => v > 0 ? v.toFixed(1) + ' см²' : '—';
  const formatLength = (v) => v > 0 ? v.toFixed(1) + ' см' : '—';

  return (
    <div className="results-table-wrapper">
      <table className="results-table">
        <thead>
          <tr>
            <th>Класс</th>
            <th>Средняя уверенность</th>
            <th>Суммарная площадь</th>
            <th>Суммарная длина</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><span className="class-badge class-leaf">🍃 Листья</span></td>
            <td>{avgConf(groups.leaf)}</td>
            <td>{formatArea(groups.leaf.area)}</td>
            <td>—</td>
          </tr>

          <tr>
            <td><span className="class-badge class-stem">🌱 Стебель</span></td>
            <td>{avgConf(groups.stem)}</td>
            <td>{formatArea(groups.stem.area)}</td>
            <td>{formatLength(groups.stem.length)}</td>
          </tr>

          <tr>
            <td><span className="class-badge class-root">🪴 Корень</span></td>
            <td>{avgConf(groups.root)}</td>
            <td>{formatArea(groups.root.area)}</td>
            <td>{formatLength(groups.root.length)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default Results;