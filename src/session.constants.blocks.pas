
{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.constants.blocks;

{$mode ObjFPC}{$H+}

interface

type

  TBlockKeys = record
    NextBlockOnNotCriterionKey : string; // BackUpBlock	: integer;
    BackUpBlockErrorsKey: string;
    MaxBlockRepetitionKey : string;
    MaxBlockRepetitionInSessionKey	: string;
    EndSessionOnHitCriterionKey	: string;
    NextBlockOnHitCriterionKey : string;
    CrtHitPorcentageKey : string;
  end;

const
  BlockKeys : TBlockKeys = (
    NextBlockOnNotCriterionKey : 'BackUpBlock'; // BackUpBlock	: integer;
    BackUpBlockErrorsKey: 'BackUpBlockErrors';
    MaxBlockRepetitionKey : 'MaxBlockRepetition';
    MaxBlockRepetitionInSessionKey	: 'MaxBlockRepetitionInSession';
    EndSessionOnHitCriterionKey	: 'EndSessionOnHitCriterion';
    NextBlockOnHitCriterionKey : 'NextBlockOnHitCriterion';
    CrtHitPorcentageKey : 'HitCriterion');

implementation

end.

