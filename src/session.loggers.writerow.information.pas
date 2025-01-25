{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.loggers.writerow.information;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, session.loggers.types;

procedure InitializeBaseHeader;
procedure Finalize;

const
  GExtention = '.info';
  GSeparator = ':';
  INFO_VERSION = '3';

var
  SaveData : TDataProcedure;
  SessionResult : string;

resourcestring
  HVERSION           = 'Version';
  HSUBJECT_NAME      = 'Nome_do_sujeito';
  HSESSION_NAME      = 'Nome_da_sessao';
  HSESSION_DESIGN    = 'Nome_do_planejamento';
  HFIRST_TIMESTAMP   = 'Primeira_timestamp';
  HBEGIN_DATE        = 'Data_Inicio';
  HBEGIN_TIME        = 'Hora_Inicio';
  HEND_DATE          = 'Data_Termino';
  HEND_TIME          = 'Hora_Termino';
  HSESSION_RESULT    = 'Resultado';
  HSESSION_ID        = 'ID';
  HGRID              = 'Grade_de_estimulos';
  HMONITOR           = 'Monitor';
  HSESSION_CANCELED  = '----------Sessao Cancelada----------';
  HTEST_MODE         = '(Modo de Teste)';
  HDURATION          = 'Duration';

implementation

uses
  session.pool,
  sdl.helpers,
  sdl.app.video.methods,
  sdl.app.grids;

var
  StartTime : TDateTime;

function Line(AName, AValue: string) : string;
begin
  Result := String.Join(GSeparator, [AName, AValue]) + LineEnding;
end;

procedure InitializeBaseHeader;
begin
  StartTime := Time;

  SaveData(
    Line(HVERSION, INFO_VERSION) +
    Line(HSUBJECT_NAME, Pool.ParticipantName) +
    Line(HSESSION_NAME, Pool.SessionName) +
    Line(HSESSION_DESIGN, Pool.DesignBasePath) +
    Line(HGRID, Grid.ToJSON) +
    Line(HMONITOR, WindowBoundsRect.ToJSON) +
    Line(HBEGIN_DATE, DateTimeToStr(Date)) +
    Line(HBEGIN_TIME, TimeToStr(StartTime)));
end;

procedure Finalize;
var
  LStopTime : TDateTime;
begin
  LStopTime := Time;
  SaveData(
    Line(HEND_DATE, DateTimeToStr(Date)) +
    Line(HEND_TIME, TimeToStr(LStopTime)) +
    Line(HDURATION, TimeToStr(LStopTime - StartTime)) +
    Line(HSESSION_RESULT, SessionResult));
end;

function MockHeader: string;
begin
  Result :=
    Line(HSUBJECT_NAME, 'Sujeito X') +
    Line(HSESSION_NAME, 'Sessão X') +
    Line(HBEGIN_DATE, DateTimeToStr(Date)) +
    Line(HBEGIN_TIME, TimeToStr(Time));
end;

end.

